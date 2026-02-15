"""Verify model predictions by showing sample-level results."""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.models.hierarchical_model import HierarchicalGrainClassifier

def main():
    # Load model
    checkpoint = 'checkpoints/exp8_heavy_oversample/best_model.pth'
    print(f"Loading: {checkpoint}")
    
    model = HierarchicalGrainClassifier(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    # Load test data
    test_dataset = GrainDatasetNew(split='test')

    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    rev_map = {v: k for k, v in label_map.items()}

    # Count correct/incorrect per class
    results = {c: {'correct': [], 'wrong': []} for c in label_map.keys()}
    
    for i in range(len(test_dataset)):
        img, labels, meta = test_dataset[i]
        true_label = meta['label']
        
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            pred = model.get_predictions(logits).item()
        
        pred_label = rev_map[pred]
        grain_id = meta['grain_id']
        
        if true_label == pred_label:
            results[true_label]['correct'].append((i, grain_id, pred_label))
        else:
            results[true_label]['wrong'].append((i, grain_id, pred_label))

    # Print summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    total_correct = 0
    total = 0
    for cls in ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']:
        correct = len(results[cls]['correct'])
        wrong = len(results[cls]['wrong'])
        total_correct += correct
        total += correct + wrong
        pct = 100 * correct / (correct + wrong) if (correct + wrong) > 0 else 0
        print(f"\n{cls}: {correct}/{correct+wrong} ({pct:.1f}%)")
        
        if wrong > 0:
            print(f"  Misclassified as:")
            wrong_preds = {}
            for idx, gid, pred in results[cls]['wrong']:
                wrong_preds[pred] = wrong_preds.get(pred, 0) + 1
            for pred, count in sorted(wrong_preds.items(), key=lambda x: -x[1]):
                print(f"    {pred}: {count}")

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    print("=" * 70)
    
    # Show specific misclassified samples for minority classes
    print("\nMisclassified minority class samples (for manual review):")
    print("-" * 70)
    for cls in ['Ooid', 'Broken ooid', 'Intraclast']:
        if results[cls]['wrong']:
            print(f"\n{cls} misclassified:")
            for idx, gid, pred in results[cls]['wrong']:
                print(f"  Sample {idx}: {gid} -> predicted as {pred}")

if __name__ == '__main__':
    main()

