"""
Combined: Staged Training + 15x Broken Ooid Oversample + TTA evaluation.

Phase 1: Freeze backbone, train all heads with oversampled batches (10 epochs)
Phase 2: Unfreeze backbone, Stage 1 only (15 epochs)
Phase 3: Freeze Stage 1, train Stage 2+3 with oversampled batches (15 epochs)
Phase 4: Fine-tune all with oversampled batches, low LR (10 epochs)
"""
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_new import GrainDatasetNew
from src.data.samplers import StageWiseBatchSampler
from src.models.hierarchical_model import HierarchicalGrainClassifier
from src.models.focal_loss import FocalLoss
from src.utils.reproducibility import set_seed


def make_oversampled_loader(batch_size, oversample, split_dir):
    ds = GrainDatasetNew(split='train', split_dir=split_dir)
    sampler = StageWiseBatchSampler(ds, batch_size=batch_size,
                                    broken_ooid_oversample=oversample)
    return DataLoader(ds, batch_sampler=sampler, num_workers=0)


def make_loader(split, batch_size, split_dir):
    ds = GrainDatasetNew(split=split, split_dir=split_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def train_epoch(model, loader, optimizer, loss_fns, device, active_stages):
    model.train()
    total_loss = 0
    for images, labels, _ in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        for k in labels: labels[k] = labels[k].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = torch.tensor(0.0, device=device)
        if 'stage1' in active_stages:
            loss = loss + loss_fns['stage1'](logits['stage1'], labels['stage1'].float())
        if 'stage2' in active_stages:
            mask = labels['stage2'] != -1
            if mask.sum() > 0:
                loss = loss + loss_fns['stage2'](logits['stage2'][mask], labels['stage2'][mask].float())
        if 'stage3' in active_stages:
            mask = labels['stage3'] != -1
            if mask.sum() > 0:
                loss = loss + loss_fns['stage3'](logits['stage3'][mask], labels['stage3'][mask].float())
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def apply_tta(image_tensor):
    augmented = [
        image_tensor,
        torch.flip(image_tensor, dims=[2]),
        torch.flip(image_tensor, dims=[1]),
        torch.rot90(image_tensor, k=1, dims=[1, 2]),
        torch.rot90(image_tensor, k=2, dims=[1, 2]),
        torch.rot90(image_tensor, k=3, dims=[1, 2]),
    ]
    return torch.stack(augmented)


def predict_logits(model, image, device, use_tta=False):
    if use_tta:
        augmented = apply_tta(image).to(device)
        logits = model(augmented)
        return {k: v.mean().reshape(1, 1) for k, v in logits.items()}
    logits = model(image.unsqueeze(0).to(device))
    return {k: v[0:1] for k, v in logits.items()}


def summarize_predictions(trues, preds):
    class_names = ['Peloid', 'Ooid', 'Broken ooid', 'Intraclast']
    trues = np.array(trues)
    preds = np.array(preds)
    cm = confusion_matrix(trues, preds, labels=[0, 1, 2, 3])
    return {
        'balanced_accuracy': float(balanced_accuracy_score(trues, preds)),
        'overall_accuracy': float((trues == preds).mean()),
        'confusion_matrix': cm.tolist(),
        'per_class_recall': {
            class_names[i]: float((preds[trues == i] == i).sum() / (trues == i).sum())
            for i in range(4)
        },
    }


def evaluate(model, loader, device):
    model.eval()
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    preds, trues = [], []
    with torch.no_grad():
        for images, _, metadata in loader:
            images = images.to(device)
            logits = model(images)
            p = model.get_predictions(logits)
            preds.extend(p.cpu().tolist())
            trues.extend([label_map[l] for l in metadata['label']])
    return balanced_accuracy_score(trues, preds)


def evaluate_dataset(model, dataset, device, use_tta=False):
    model.eval()
    label_map = {'Peloid': 0, 'Ooid': 1, 'Broken ooid': 2, 'Intraclast': 3}
    preds, trues = [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, _, metadata = dataset[idx]
            logits = predict_logits(model, image, device, use_tta=use_tta)
            pred = model.get_predictions(logits).item()
            preds.append(pred)
            trues.append(label_map[metadata['label']])
    return summarize_predictions(trues, preds)


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Split dir: {args.split_dir}")

    val_loader = make_loader('val', args.batch_size, args.split_dir)
    model = HierarchicalGrainClassifier(pretrained=True).to(device)

    loss_fns = {
        'stage1': FocalLoss(alpha=0.25, gamma=2.0),
        'stage2': FocalLoss(alpha=0.50, gamma=2.0),
        'stage3': FocalLoss(alpha=0.75, gamma=2.0),
    }

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ba = 0.0
    history = []

    def run_phase(phase_num, epochs, active_stages, lr, freeze_fn=None, unfreeze_fn=None):
        nonlocal best_ba
        print(f"\n{'='*60}\nPHASE {phase_num}: stages={active_stages}, lr={lr}")

        if freeze_fn:   freeze_fn()
        if unfreeze_fn: unfreeze_fn()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        train_loader = make_oversampled_loader(args.batch_size, args.oversample, args.split_dir)

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, loss_fns, device, active_stages)
            ba = evaluate(model, val_loader, device)
            print(f"  Phase {phase_num} Epoch {epoch:2d}: loss={loss:.4f}  val_ba={ba:.4f}")
            history.append({'phase': phase_num, 'epoch': epoch, 'loss': loss, 'val_ba': ba})
            if ba > best_ba:
                best_ba = ba
                torch.save(model.state_dict(), ckpt_dir / 'best_model.pth')

    # Phase 1: frozen backbone, all heads
    run_phase(1, args.phase1_epochs, ['stage1','stage2','stage3'], lr=1e-3,
              freeze_fn=model.freeze_backbone)

    # Phase 2: unfreeze backbone, Stage 1 only
    def freeze_heads_23():
        model.unfreeze_backbone()
        for p in model.head_stage2.parameters(): p.requires_grad = False
        for p in model.head_stage3.parameters(): p.requires_grad = False

    run_phase(2, args.phase2_epochs, ['stage1'], lr=1e-4,
              unfreeze_fn=freeze_heads_23)

    # Phase 3: freeze Stage 1, train Stage 2+3
    def freeze_head1_unfreeze_23():
        for p in model.head_stage1.parameters(): p.requires_grad = False
        for p in model.head_stage2.parameters(): p.requires_grad = True
        for p in model.head_stage3.parameters(): p.requires_grad = True

    run_phase(3, args.phase3_epochs, ['stage2','stage3'], lr=1e-4,
              unfreeze_fn=freeze_head1_unfreeze_23)

    # Phase 4: fine-tune all
    def unfreeze_all():
        for p in model.parameters(): p.requires_grad = True

    run_phase(4, args.phase4_epochs, ['stage1','stage2','stage3'], lr=1e-5,
              unfreeze_fn=unfreeze_all)

    torch.save(model.state_dict(), ckpt_dir / 'final_model.pth')
    model.load_state_dict(torch.load(ckpt_dir / 'best_model.pth', map_location=device))
    test_dataset = GrainDatasetNew(split='test', split_dir=args.split_dir)
    test_no_tta = evaluate_dataset(model, test_dataset, device, use_tta=False)
    test_tta = evaluate_dataset(model, test_dataset, device, use_tta=True)

    with open(ckpt_dir / 'training_history.json', 'w') as f:
        json.dump({'history': history, 'best_val_ba': best_ba,
                   'oversample': args.oversample, 'seed': args.seed,
                   'split_dir': args.split_dir, 'test_without_tta': test_no_tta,
                   'test_with_tta': test_tta}, f, indent=2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'seed': args.seed,
            'split_dir': args.split_dir,
            'checkpoint_dir': str(ckpt_dir),
            'best_val_ba': best_ba,
            'test_without_tta': test_no_tta,
            'test_with_tta': test_tta,
            'history': history,
        }, f, indent=2)

    print(f"\nDone. Best val balanced accuracy: {best_ba:.4f}")
    print(f"Test BA without TTA: {test_no_tta['balanced_accuracy']:.4f}")
    print(f"Test BA with TTA:    {test_tta['balanced_accuracy']:.4f}")
    print(f"Saved results to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--oversample',     type=int,   default=15)
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--split_dir',      type=str,   default='data/processed')
    parser.add_argument('--output',         type=str,   default='results/exp_combined_results.json')
    parser.add_argument('--phase1_epochs',  type=int,   default=10)
    parser.add_argument('--phase2_epochs',  type=int,   default=15)
    parser.add_argument('--phase3_epochs',  type=int,   default=15)
    parser.add_argument('--phase4_epochs',  type=int,   default=10)
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints/exp_combined')
    args = parser.parse_args()
    main(args)
