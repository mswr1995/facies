# Hierarchical grain facies classification in carbonate thin sections using deep learning

12-minute seminar structure for a sedimentology forum.

## Core Message

Rare carbonate grain facies can carry important depositional information, but they are exactly the classes most likely to be missed by generic computer-vision models because the dataset is highly imbalanced and the diagnostic features are subtle. The proposed solution is to make the classifier follow sedimentological reasoning: first separate peloids, then distinguish ooid-like grains from intraclasts, then separate intact and broken ooids.

## Slide 1: Title

**Slide text**

Hierarchical grain facies classification in carbonate thin sections using deep learning

Musawer Muradi  
Advisors: Assoc. Prof. Alev Mutlu, Asst. Prof. Arnaud Bruno Cedric Gallois  
Kocaeli University

**Talking points**

This talk is about using deep learning for carbonate grain facies classification, but the main idea is geological: the model is structured around how sedimentologists distinguish grains.

## Slide 2: Why Carbonate Grain Classification Matters

**Slide text**

- Carbonate grains record depositional energy, reworking, and diagenetic history.
- Grain assemblages support palaeoenvironmental interpretation.
- Rare components, such as broken ooids and intraclasts, can be disproportionately informative.

**Talking points**

Carbonate thin sections are not just images. Their grain types preserve information about depositional processes. Broken ooids can indicate high-energy reworking, while intraclasts can indicate erosion and redeposition of earlier carbonate material.

## Slide 3: Thin Sections and Grain Facies

**Slide text**

- Samples are observed in plain polarised light.
- Each grain is manually delineated as a polygon.
- The model receives masked grain patches, not whole images.

**Suggested visual**

Use article figure: `figs/fig1_pipeline.pdf`, or a simplified version later in the PowerPoint.

**Talking points**

The task is grain-level classification. We are not asking whether a whole image belongs to one facies. Each annotated grain is cropped, masked, and classified individually.

## Slide 4: Why Manual Classification Is Hard

**Slide text**

- Requires expert petrographic interpretation.
- Diagenesis can erase diagnostic textures.
- Peloids, altered ooids, and intraclasts may overlap visually.
- Reproducibility is difficult when grains are ambiguous.

**Talking points**

The difficult cases are not random. They occur where geological processes have genuinely blurred the diagnostic boundaries between classes, especially through micritisation.

## Slide 5: Dataset

**Slide text**

- 18 PPL micrographs
- 9 thin sections
- 2,642 annotated grains
- 4 target classes: peloid, ooid, broken ooid, intraclast
- Excluded from modelling: ostracod, bivalve fragment, quartz grain due to very small counts

**Suggested visual**

`seminar_graphs/seminar_pipeline.png`

**Talking points**

The dataset is small by computer-vision standards but realistic for expert petrographic annotation. Labels were produced at grain level by an expert sedimentologist.

## Slide 6: The Core Problem: Extreme Imbalance

**Slide text**

- Peloid: 2,300 grains
- Ooid: 210 grains
- Intraclast: 103 grains
- Broken ooid: 29 grains
- Peloid-to-broken-ooid ratio: about 79:1

**Suggested visual**

`seminar_graphs/class_distribution.png`

**Talking points**

If a model simply learns to prefer peloid, it can obtain high ordinary accuracy without solving the scientific problem. That is why Balanced Accuracy and per-class recall are more important here than overall accuracy.

## Slide 7: Grain Classes and Visual Ambiguity

**Slide text**

- Peloid: micritic, often structureless.
- Ooid: concentric cortical lamination.
- Broken ooid: fractured cortex.
- Intraclast: reworked carbonate grain, sometimes angular or internally structured.

**Suggested visual**

Use article figure: `figs/fig2_sample_patches.pdf`.

**Talking points**

The visual difference is often small. The broken-ooid signal may be only part of the grain boundary. The intraclast signal may be weak remnant fabric or angularity. These are not like separating cats from cars.

## Slide 8: Why General Computer Vision Is Not Enough

**Slide text**

- Generic image classifiers assume visually separable classes and many examples.
- Carbonate grains violate both assumptions.
- SAM plus hand-crafted features and XGBoost was useful as a pilot.
- But generic segmentation plus geometry/texture descriptors did not fully capture subtle petrographic distinctions.
- The model needs to learn grain-level texture while respecting geological hierarchy.

**Talking points**

We first explored a conventional computer-vision route: segment grains, extract geometric and textural features, and classify them with XGBoost. That exposed the limitation. The difficult part is not only finding grains; it is separating subtle, rare, geologically defined classes under extreme imbalance.

## Slide 9: Hierarchical Classification Idea

**Slide text**

Instead of one flat four-class decision:

1. Is this grain a peloid?
2. If not, is it ooid-like or an intraclast?
3. If ooid-like, is it intact or broken?

**Suggested visual**

`seminar_graphs/hierarchical_decision_tree.png`

**Talking points**

The hierarchy follows the sedimentological decision process. It also prevents the rare broken-ooid decision from being trained directly against thousands of peloids.

## Slide 10: Model and Training Strategy

**Slide text**

- ResNet-18 backbone extracts features from each masked grain patch.
- Three binary heads implement the hierarchy.
- Stage-specific Focal Loss addresses local imbalance.
- Broken ooids are oversampled during training.
- Staged training reduces early dominance by the peloid decision.
- Six-orientation averaging stabilises inference.

**Suggested visual**

Use a simplified architecture diagram, or combine `seminar_graphs/hierarchical_decision_tree.png` with one pipeline sentence.

**Talking points**

The important point is not the specific neural-network brand. The important point is that the training is imbalance-aware at every step: architecture, loss, sampling, and training schedule.

## Slide 11: Results: What Actually Helped?

**Slide text**

- Hierarchical base model recovered only 1/6 broken ooids.
- Targeted oversampling increased this to 4/6.
- Full model recovered 5/6 broken ooids.
- Balanced Accuracy increased from 63.3% to 83.5%.

**Suggested visuals**

`seminar_graphs/broken_ooid_recall_ablation.png`  
`seminar_graphs/ablation_balanced_accuracy.png`

**Talking points**

The most important result is the ablation. Architecture alone was not enough. Oversampling provided the rare-class gradient signal, and staged training improved the balance between classes.

## Slide 12: Errors, Geological Interpretation, and Take-Home Message

**Slide text**

- Full model: 83.5% Balanced Accuracy and 92.1% Overall Accuracy.
- Main error: peloids predicted as intraclasts.
- This reflects a real petrographic ambiguity under advanced micritisation.
- Take-home: petrographic AI should encode geological structure, not only generic image-recognition machinery.

**Suggested visuals**

`seminar_graphs/confusion_matrix_full_model.png`  
Article figure: `figs/fig6_misclassifications.pdf`

**Talking points**

The remaining mistakes are interpretable. Peloid-intraclast confusion is exactly where human interpretation also becomes uncertain. The model should therefore be viewed as a decision-support tool and a proof of concept, not as a final replacement for expert petrography.

## Optional Backup Slides

### Backup 1: Test Set Uncertainty

**Suggested visual**

`seminar_graphs/test_set_distribution.png`

**Use if asked**

The broken-ooid test set contains only six examples, so any class-level percentage has a wide confidence interval.

### Backup 2: Flat Baseline Comparison

**Suggested visual**

`seminar_graphs/flat_vs_hierarchical_recall.png`

**Use if asked**

The flat ResNet baseline has higher Balanced Accuracy on this split, but its broken-ooid advantage depends on one extra correct prediction among only six test grains. The hierarchical model has stronger intraclast recall and better geological interpretability.

### Backup 3: Latent Space

**Suggested visual**

Article figure: `figs/fig5_tsne.pdf`

**Use if asked**

The feature space separates peloids more clearly than the non-peloid classes, which supports the staged hierarchy.

## Timing

- Slides 1-2: 1.5 min
- Slides 3-4: 2 min
- Slides 5-7: 2.5 min
- Slides 8-10: 3 min
- Slides 11-12: 3 min

## One-Sentence Closing

This work shows that carbonate grain facies classification should not be treated as generic image classification: the model performs best when its structure reflects sedimentological hierarchy and the training process explicitly protects rare but meaningful classes.
