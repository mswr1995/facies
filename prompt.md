You are acting as a senior machine-learning engineer helping me implement a carbonate grain classification pipeline.
You have access to my current codebase and the latest experiment results (ResNet-18 hierarchical binary model, Fold-0 results already reviewed).

Context:

Task: object-level classification of carbonate grains

Classes: peloid, ooid, broken ooid, intraclast

Extreme class imbalance (broken ooids ≈ 1%)

Pixel-level classification is NOT used

Each sample is a grain-centered image patch (96×96 RGB) with a binary mask

Masking strategy: image multiplied by mask (outside grain = 0)

Model design (must follow exactly):

Backbone: ResNet-18 pretrained on ImageNet

One shared backbone

Three hierarchical binary heads:

Peloid vs Non-peloid (applied to all grains)

Ooid-like (whole + broken) vs Intraclast (applied only to non-peloids)

Whole vs Broken ooid (applied only to ooid-like grains)

Each head architecture:

Linear(512 → 128) + ReLU + Dropout(0.3) + Linear(128 → 1)

Sigmoid output

Training requirements:

Framework: PyTorch

Optimizer: AdamW, lr = 1e-4, weight_decay = 1e-4

Loss: Focal Loss

Stage 1: alpha = 0.25, gamma = 2.0

Stage 2: alpha = 0.50, gamma = 2.0

Stage 3: alpha = 0.75, gamma = 2.0

Use class-aware sampling:

Stage 1: balance peloid vs non-peloid per batch

Stage 2: balance ooid-like vs intraclast

Stage 3: heavily oversample broken ooids

Dataset splitting is done by image, not by grain

Important design decision (based on latest results):

Broken ooid classification is unreliable due to scarcity

Stage-3 output must be treated as a probability/confidence estimate

Threshold for broken ooid must be conservative and configurable

Do NOT optimize stage-3 for accuracy

Inference logic must be explicit in code:

if P(peloid) > T1:
label = "peloid"
else:
if P(ooid_like) > T2:
if P(broken) > T3:
label = "broken_ooid"
else:
label = "ooid"
else:
label = "intraclast"

What I want you to generate (step by step, not all at once):

Dataset class that loads masked grain patches

Class-aware sampler for hierarchical training

Model definition (shared backbone + 3 heads)

Focal loss implementation

Training loop respecting hierarchical logic

Inference function using the decision tree above

Evaluation utilities reporting precision, recall, PR-AUC per stage

Constraints:

Do NOT use pixel-level segmentation

Do NOT use multi-class softmax

Do NOT assume balanced data

Do NOT over-engineer

Code must be clean, readable, and suitable for research

Generate correct, production-quality PyTorch code that integrates cleanly with the existing codebase.