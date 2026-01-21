"""
Labelme annotation loader for carbonate grain classification.
Handles JSON parsing, filtering, and basic grain extraction.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import base64


# Valid grain classes (excludes rare bioclasts and minerals)
VALID_CLASSES = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']

# Class name normalization (handle case variations)
CLASS_MAPPING = {
    'peloid': 'Peloid',
    'Peloid': 'Peloid',
    'ooid': 'Ooid',
    'Ooid': 'Ooid',
    'broken ooid': 'Broken ooid',
    'Broken ooid': 'Broken ooid',
    'Broken Ooid': 'Broken ooid',
    'intraclast': 'Intraclast',
    'Intraclast': 'Intraclast',
}


def load_labelme_json(json_path: Path) -> Dict:
    """Load and parse a labelme JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_image_from_labelme(json_data: Dict, image_dir: Path) -> np.ndarray:
    """
    Load the corresponding image for a labelme annotation.
    
    First tries to load from embedded base64 imageData.
    Falls back to loading from file path if imageData is not present.
    
    Args:
        json_data: Parsed labelme JSON
        image_dir: Directory containing images (used as fallback)
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    # Try loading from embedded base64 data first
    if 'imageData' in json_data and json_data['imageData']:
        try:
            image_data = base64.b64decode(json_data['imageData'])
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Warning: Failed to decode embedded image data: {e}")
            # Fall through to file loading
    
    # Fallback: load from file
    image_path = image_dir / json_data['imagePath']
    
    # Try case-insensitive search if exact match fails
    if not image_path.exists():
        # Look for file with different case
        for img_file in image_dir.glob('*'):
            if img_file.name.lower() == json_data['imagePath'].lower():
                image_path = img_file
                break
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load as RGB (OpenCV loads as BGR, so convert)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def polygon_to_mask(points: List[List[float]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert polygon points to binary mask.
    
    Args:
        points: List of [x, y] coordinates
        image_shape: (height, width) of the image
        
    Returns:
        Binary mask (H, W) with 1 inside polygon, 0 outside
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    return mask


def compute_centroid(points: List[List[float]]) -> Tuple[int, int]:
    """
    Compute centroid of a polygon.
    
    Returns:
        (cx, cy) in pixel coordinates
    """
    points_array = np.array(points)
    cx = int(np.mean(points_array[:, 0]))
    cy = int(np.mean(points_array[:, 1]))
    return cx, cy


def extract_grains_from_labelme(
    json_path: Path,
    image_dir: Path,
    filter_classes: bool = True
) -> List[Dict]:
    """
    Extract all grain annotations from a labelme JSON file.
    
    Args:
        json_path: Path to labelme JSON file
        image_dir: Directory containing the image
        filter_classes: If True, exclude rare classes (bivalve, ostracod, quartz)
        
    Returns:
        List of grain dictionaries with keys:
            - label: class name
            - points: polygon coordinates
            - centroid: (cx, cy)
            - mask: binary mask (optional, computed on demand)
            - image_name: source image filename
    """
    data = load_labelme_json(json_path)
    image_name = json_path.stem
    image = load_image_from_labelme(data, image_dir)
    image_shape = image.shape[:2]  # (H, W)
    
    grains = []
    
    for shape in data['shapes']:
        label = shape['label']
        
        # Normalize label
        label = CLASS_MAPPING.get(label, label)
        
        # Filter out invalid classes
        if filter_classes and label not in VALID_CLASSES:
            continue
        
        points = shape['points']
        centroid = compute_centroid(points)
        
        grain_info = {
            'label': label,
            'points': points,
            'centroid': centroid,
            'image_name': image_name,
            'image_shape': image_shape,
        }
        
        grains.append(grain_info)
    
    return grains


def load_all_annotations(
    data_dir: Path,
    filter_classes: bool = True
) -> Dict[str, List[Dict]]:
    """
    Load all labelme annotations from a directory.
    
    Args:
        data_dir: Directory containing both JSON and image files
        filter_classes: If True, exclude rare classes
        
    Returns:
        Dictionary mapping image_name -> list of grains
    """
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob('*.json'))
    
    all_annotations = {}
    
    for json_path in json_files:
        grains = extract_grains_from_labelme(json_path, data_dir, filter_classes)
        image_name = json_path.stem
        all_annotations[image_name] = grains
    
    return all_annotations


def get_class_statistics(annotations: Dict[str, List[Dict]]) -> Dict:
    """
    Compute class distribution statistics.
    
    Returns:
        Dictionary with class counts and percentages
    """
    class_counts = {}
    
    for image_name, grains in annotations.items():
        for grain in grains:
            label = grain['label']
            class_counts[label] = class_counts.get(label, 0) + 1
    
    total = sum(class_counts.values())
    
    stats = {
        'counts': class_counts,
        'total': total,
        'percentages': {cls: (count / total * 100) for cls, count in class_counts.items()}
    }
    
    return stats


def extract_grain_patch(
    image: np.ndarray,
    grain_info: Dict,
    patch_size: int = 96,
    with_mask: bool = True
) -> np.ndarray:
    """
    Extract grain patch centered on centroid.
    
    Args:
        image: Full image (H, W, 3)
        grain_info: Grain dictionary with 'centroid' and 'points' keys
        patch_size: Target size for square patch (default 96)
        with_mask: If True, multiply by grain mask (zeros outside)
        
    Returns:
        Patch of size (patch_size, patch_size, 3)
    """
    cx, cy = grain_info['centroid']
    h, w = image.shape[:2]
    
    # Calculate crop bounds centered on centroid
    half_size = patch_size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)
    
    # Crop patch
    patch = image[y1:y2, x1:x2].copy()
    
    # Apply mask if requested
    if with_mask:
        mask = polygon_to_mask(grain_info['points'], image.shape[:2])
        mask_crop = mask[y1:y2, x1:x2]
        patch = patch * mask_crop[:, :, np.newaxis]
    
    # Pad if necessary (near image edges)
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        pad_h = max(0, patch_size - patch.shape[0])
        pad_w = max(0, patch_size - patch.shape[1])
        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    # Crop to exact size if larger
    patch = patch[:patch_size, :patch_size]
    
    return patch
