"""
Extract CNN embeddings from trained hierarchical model.

Freezes the backbone and extracts 512-D feature vectors for all grains.
These embeddings are used as input to XGBoost for Stage 3 classification.
"""
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import GrainDataset
from src.models.hierarchical_model import HierarchicalGrainClassifier


def extract_embeddings(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu'
) -> dict:
    """
    Extract CNN embeddings for all samples in a data loader.
    
    Args:
        model: Trained hierarchical model
        data_loader: DataLoader for dataset
        device: Device to run on
        
    Returns:
        Dictionary mapping grain_id to embedding vector
    """
    model.eval()
    model.to(device)
    
    embeddings_dict = {}
    metadata_dict = {}
    
    print(f"Extracting embeddings from {len(data_loader.dataset)} samples...")
    
    with torch.no_grad():
        for images, labels, metadata in tqdm(data_loader):
            images = images.to(device)
            
            # Extract features from backbone
            features = model.backbone(images)
            features = torch.flatten(features, 1)
            
            # Store embeddings
            features_np = features.cpu().numpy()
            
            for i in range(len(features_np)):
                # Properly extract grain_id as integer
                grain_id_value = metadata['grain_id'][i]
                if torch.is_tensor(grain_id_value):
                    grain_id_value = grain_id_value.item()
                grain_id = str(int(grain_id_value))  # Convert to string for JSON
                
                embeddings_dict[grain_id] = features_np[i].tolist()
                
                metadata_dict[grain_id] = {
                    'label': metadata['label'][i],
                    'image_name': metadata['image_name'][i],
                    'patch_filename': metadata['patch_filename'][i]
                }
    
    return embeddings_dict, metadata_dict


def filter_ooid_like_grains(
    embeddings_dict: dict,
    metadata_dict: dict
) -> dict:
    """
    Filter to keep only ooid-like grains (Ooid + Broken ooid).
    
    Args:
        embeddings_dict: Dictionary of embeddings
        metadata_dict: Dictionary of metadata
        
    Returns:
        Filtered embeddings dictionary
    """
    ooid_like_embeddings = {}
    ooid_like_metadata = {}
    
    for grain_id in embeddings_dict.keys():
        label = metadata_dict[grain_id]['label']
        if label in ['Ooid', 'Broken ooid']:
            ooid_like_embeddings[grain_id] = embeddings_dict[grain_id]
            ooid_like_metadata[grain_id] = metadata_dict[grain_id]
    
    print(f"\nFiltered to {len(ooid_like_embeddings)} ooid-like grains")
    print(f"  Ooid: {sum(1 for m in ooid_like_metadata.values() if m['label'] == 'Ooid')}")
    print(f"  Broken ooid: {sum(1 for m in ooid_like_metadata.values() if m['label'] == 'Broken ooid')}")
    
    return ooid_like_embeddings, ooid_like_metadata


def save_embeddings(
    embeddings_dict: dict,
    metadata_dict: dict,
    output_path: str
):
    """
    Save embeddings and metadata to JSON file.
    
    Args:
        embeddings_dict: Dictionary of embeddings
        metadata_dict: Dictionary of metadata
        output_path: Where to save
    """
    output_data = {
        'embeddings': embeddings_dict,
        'metadata': metadata_dict
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEmbeddings saved to: {output_path}")


def main():
    """Extract embeddings from trained model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CNN embeddings')
    parser.add_argument('--fold', type=int, required=True, help='Fold number')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                       help='Which split to extract embeddings for')
    parser.add_argument('--ooid-only', action='store_true',
                       help='Filter to ooid-like grains only')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: data/processed/embeddings_fold{N}_{split}.json)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = HierarchicalGrainClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Freeze backbone
    model.freeze_backbone()
    print("Backbone frozen for embedding extraction")
    
    # Create dataset
    metadata_path = f'data/processed/fold_{args.fold}_metadata.json'
    patches_dir = 'data/processed/patches'
    
    dataset = GrainDataset(
        metadata_path=metadata_path,
        patches_dir=patches_dir,
        split=args.split
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nExtracting embeddings for {args.split} split")
    print(f"  Total samples: {len(dataset)}")
    
    # Extract embeddings
    embeddings_dict, metadata_dict = extract_embeddings(model, data_loader, device)
    
    # Filter to ooid-like if requested
    if args.ooid_only:
        embeddings_dict, metadata_dict = filter_ooid_like_grains(
            embeddings_dict, metadata_dict
        )
    
    # Determine output path
    if args.output is None:
        suffix = '_ooid_only' if args.ooid_only else ''
        args.output = f'data/processed/embeddings_fold{args.fold}_{args.split}{suffix}.json'
    
    # Save
    save_embeddings(embeddings_dict, metadata_dict, args.output)
    
    print(f"\n✅ Embedding extraction complete!")
    print(f"   Extracted {len(embeddings_dict)} embeddings")
    print(f"   Embedding dimension: 512")


if __name__ == '__main__':
    main()
