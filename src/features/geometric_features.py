"""
Geometric feature extraction for grain analysis.

Minimal set of shape-based features to complement CNN embeddings.
"""
import numpy as np
import cv2
from typing import Dict, List
from pathlib import Path
import json


def compute_grain_features(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute geometric features from a grain mask.
    
    Args:
        mask: Binary mask (H, W) with 1=grain, 0=background
        
    Returns:
        Dictionary of geometric features
    """
    # Ensure binary
    mask = (mask > 0).astype(np.uint8)
    
    # Find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Return zeros if no contour found
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'equivalent_diameter': 0.0,
            'circularity': 0.0,
            'solidity': 0.0,
            'aspect_ratio': 0.0,
            'extent': 0.0
        }
    
    # Use largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Basic measurements
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Equivalent diameter (diameter of circle with same area)
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0
    
    # Circularity (4π × area / perimeter²)
    # 1.0 = perfect circle, <1.0 = irregular
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
    circularity = min(circularity, 1.0)  # Cap at 1.0 due to numerical precision
    
    # Convex hull for solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    
    # Bounding rectangle for aspect ratio and extent
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'equivalent_diameter': float(equivalent_diameter),
        'circularity': float(circularity),
        'solidity': float(solidity),
        'aspect_ratio': float(aspect_ratio),
        'extent': float(extent)
    }


def compute_intensity_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute intensity-based features from masked grain image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        
    Returns:
        Dictionary of intensity features
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Extract grain pixels
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return {
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'intensity_variance': 0.0
        }
    
    grain_pixels = gray[mask_bool]
    
    mean_intensity = float(np.mean(grain_pixels))
    std_intensity = float(np.std(grain_pixels))
    intensity_variance = float(np.var(grain_pixels))
    
    return {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'intensity_variance': intensity_variance
    }


def extract_features_for_grain(
    image: np.ndarray,
    mask: np.ndarray,
    include_intensity: bool = True
) -> Dict[str, float]:
    """
    Extract all features for a single grain.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        include_intensity: Whether to include intensity features
        
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Geometric features
    geom_features = compute_grain_features(mask)
    features.update(geom_features)
    
    # Intensity features (optional)
    if include_intensity:
        intensity_features = compute_intensity_features(image, mask)
        features.update(intensity_features)
    
    return features


def load_grain_with_mask(
    grain_metadata: Dict,
    patches_dir: Path,
    raw_data_dir: Path
) -> tuple:
    """
    Load grain patch and reconstruct its mask.
    
    Args:
        grain_metadata: Grain metadata dictionary
        patches_dir: Directory containing patch images
        raw_data_dir: Directory containing original labelme annotations
        
    Returns:
        (image, mask) tuple
    """
    from ..data.labelme_loader import (
        load_labelme_json,
        load_image_from_labelme,
        polygon_to_mask
    )
    
    # Load patch image
    patch_path = patches_dir / grain_metadata['patch_filename']
    image = cv2.imread(str(patch_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load original annotation to get polygon
    json_path = raw_data_dir / f"{grain_metadata['image_name']}.json"
    json_data = load_labelme_json(json_path)
    
    # Find the grain's polygon
    grain_idx = grain_metadata['grain_idx']
    grain_shape = json_data['shapes'][grain_idx]
    points = grain_shape['points']
    
    # Get original image shape
    original_image = load_image_from_labelme(json_data, raw_data_dir)
    image_shape = original_image.shape[:2]
    
    # Create full-size mask
    full_mask = polygon_to_mask(points, image_shape)
    
    # Crop mask to match patch
    cx, cy = grain_metadata['centroid']
    patch_size = image.shape[0]  # Assuming square
    half_size = patch_size // 2
    
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(image_shape[1], cx + half_size)
    y2 = min(image_shape[0], cy + half_size)
    
    mask_crop = full_mask[y1:y2, x1:x2]
    
    # Pad if necessary
    if mask_crop.shape[0] < patch_size or mask_crop.shape[1] < patch_size:
        pad_h = max(0, patch_size - mask_crop.shape[0])
        pad_w = max(0, patch_size - mask_crop.shape[1])
        mask_crop = np.pad(mask_crop, ((0, pad_h), (0, pad_w)), mode='constant')
    
    # Crop to exact size
    mask = mask_crop[:patch_size, :patch_size]
    
    return image, mask


def extract_features_for_dataset(
    metadata_path: str,
    patches_dir: str,
    raw_data_dir: str,
    output_path: str = None,
    include_intensity: bool = True
):
    """
    Extract geometric features for all grains in dataset.
    
    Args:
        metadata_path: Path to grain metadata JSON
        patches_dir: Directory containing patch images
        raw_data_dir: Directory containing raw labelme annotations
        output_path: Where to save features (optional)
        include_intensity: Whether to include intensity features
        
    Returns:
        Dictionary mapping grain_id to features
    """
    from tqdm import tqdm
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    patches_dir = Path(patches_dir)
    raw_data_dir = Path(raw_data_dir)
    
    all_features = {}
    
    print(f"\nExtracting features for {len(metadata)} grains...")
    
    for grain_meta in tqdm(metadata):
        grain_id = grain_meta['grain_id']
        
        try:
            # Load grain image and mask
            image, mask = load_grain_with_mask(grain_meta, patches_dir, raw_data_dir)
            
            # Extract features
            features = extract_features_for_grain(image, mask, include_intensity)
            
            # Add metadata
            features['grain_id'] = grain_id
            features['label'] = grain_meta['label']
            features['image_name'] = grain_meta['image_name']
            
            all_features[grain_id] = features
            
        except Exception as e:
            print(f"\nWarning: Failed to extract features for grain {grain_id}: {e}")
            continue
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_features, f, indent=2)
        
        print(f"Features saved to: {output_path}")
    
    return all_features


if __name__ == '__main__':
    """Test feature extraction on a sample grain."""
    import sys
    sys.path.append('.')
    
    # Create synthetic test data
    print("Testing geometric feature extraction...")
    
    # Circle
    circle_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(circle_mask, (50, 50), 30, 1, -1)
    
    circle_features = compute_grain_features(circle_mask)
    print("\nCircle features:")
    for key, value in circle_features.items():
        print(f"  {key:25s}: {value:.4f}")
    
    # Rectangle
    rect_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(rect_mask, (30, 40), (70, 60), 1, -1)
    
    rect_features = compute_grain_features(rect_mask)
    print("\nRectangle features:")
    for key, value in rect_features.items():
        print(f"  {key:25s}: {value:.4f}")
    
    print("\n✅ Feature extraction test complete!")
