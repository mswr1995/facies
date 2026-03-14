# Research Article Outline: Mathematical & Methodological Framework for Carbonate Microfacies Classification

This document serves as a structured guide and repository of ideas, mathematical formulations, and critical technical points to lead your writing process.

## 1. Introduction
*   **The Geological Problem:** Introduce carbonate microfacies classification (Peloids, Ooids, Intraclasts). Emphasize the challenge of manual petrography (subjectivity, time).
*   **The Computational Challenge:** Highlight the central problem: **Extreme Class Imbalance** ($N_{peloid} \gg N_{broken\_ooid}$). Explain why standard Deep Learning fails (majority class dominance limits gradients for minority features).
*   **Our Contribution:** Propose a novel system combining a **Hierarchical Decision Structure** with **Dynamic Loss Function Engineering (Focal Loss)** and **Staged Sampling**. 

## 2. Methodology & Mathematical Framework
*This is the core technical section of the paper. Focus on why the methods work mathematically, not just that you used them.*

### 2.1 Problem Formulation & Dataset Representation
*   Define the dataset mathematically. Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be the set of $N$ grains.
*   Let $x_i \in \mathbb{R}^{C \times H \times W}$ be the masked input patch.
*   Let $y_i \in \{1, 2, 3, 4\}$ be the flat class labels. Provide the severe probability distribution $P(Y=y)$.

### 2.2 Re-framing the Problem: Hierarchical Decomposition
*   **Rationale:** Flat softmax fails due to overlapping feature spaces and imbalance. Explain the transition to conditional probabilities.
*   Define the binary classification hierarchy. Instead of predicting $P(Y=y | X)$, we predict a sequence:
    1.  **Stage 1:** $P(Peloid | X)$
    2.  **Stage 2:** $P(OoidLike | X, \neg Peloid)$
    3.  **Stage 3:** $P(Broken | X, OoidLike)$
*   **Mathematical Justification:** Using the chain rule of probability, show how the final class probability is constructed. For example, the probability of a grain being an Intraclast is:
    $$ P(Intraclast) = (1 - P(Peloid)) \times (1 - P(OoidLike)) $$

### 2.3 Feature Extraction (The Backbone)
*   Briefly describe the ResNet-18 architecture ($f_\theta(x_i)$ map to $\mathbb{R}^d$ where $d=512$).
*   Explain the branching: The same latent representation $h_i$ is fed into three independent Multi-Layer Perceptrons ($MLP^{(s)}$ for stage $s \in \{1,2,3\}$).
    $$ h_i = f_\theta(x_i) $$
    $$ p_i^{(s)} = \sigma(MLP^{(s)}(h_i)) $$ where $\sigma$ is the sigmoid function.

### 2.4 Addressing Imbalance: Custom Loss Functions
*   **The Baseline Failure:** Briefly show standard Cross Entropy (CE) loss and explain how the gradients $\frac{\partial \mathcal{L}_{CE}}{\partial \theta}$ are overwhelmed by the majority class.
*   **The Solution: Stage-Wise Focal Loss:**
    *   Introduce the Focal Loss formulation:
        $$ \mathcal{L}_{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$
    *   **Explain parameters mathematically:** 
        *   **Focusing parameter ($\gamma$):** Explain how $(1-p_t)^\gamma$ reduces the relative loss for well-classified examples ($p_t > 0.5$), forcing the network gradients to update based on hard, misclassified examples.
        *   **Weighting factor ($\alpha_t$):** Explain how $\alpha$ counteracts class imbalance.
    *   **Crucial Detail from your experiments:** Detail how varying $\alpha$ across the hierarchy stages (e.g., Stage 1 vs. Stage 3) was necessary because the local imbalance severity changes at each node. Include the derivative equation you corrected in your early experiments to demonstrate technical rigor.

### 2.5 Regularization via Sampling and Augmentation
*   **Staged Training & Oversampling:** Explain mathematically why oversampling was needed *only* at Stage 3. It artificially alters the prior probability $P(\text{Broken})$ seen during training to prevent parameter collapse in $MLP^{(3)}$.
*   **Test-Time Augmentation (TTA):** Formulate TTA as an expectation over transformations. Let $\mathcal{T}$ be a set of $K$ valid geometric transformations (rotations/flips). The final predicted probability is:
    $$ \hat{p_i} = \frac{1}{K} \sum_{k=1}^K p(y | T_k(x_i), \theta) $$
    Discuss how this reduces prediction variance for geometrically ambiguous grains.

## 3. Experimental Setup & Evaluation Metrics
*   **Evaluation Metrics:** State why Overall Accuracy is a flawed metric for this dataset. Define **Precision, Recall, F1-Score, and Balanced Accuracy**. Provide the formulas to ground the paper technically.
    *   $$ Balanced Accuracy = \frac{Sensitivity + Specificity}{2} $$
*   **Cross-Validation Strategy:** Explain the Strict Stratified Group K-Fold split (Group = Source Image). Emphasize that preventing image-level leakage is critical for valid results.

## 4. Results & Ablation Study
*This section justifies the architecture choices.*
*   **Baseline vs. Focal Loss:** Show the mathematical/empirical failure of the flat RF/XGB models without focal scaling compared to the first iteration of your focal loss model.
*   **Hierarchical vs. Flat Deep Learning:** Compare the flat ResNet baseline to your hierarchical ResNet. Focus on the gain in minority class recall.
*   **Ablation of Components:** Provide a table tracking performance as components are added (ResNet -> +Class Weights -> +Staged Oversample -> +TTA). This proves that every mathematical piece of your methodology acts on the data as intended. (Reference your `COMPARISON_TABLE.md`).

## 5. Discussion
*   **Why did Flat Architectures fail?** Discuss the overlap in feature space (e.g., why $f_\theta$ mapped Broken Ooids close to Intraclasts in the latent space without hierarchical forcing).
*   **The Role of TTA:** Discuss how TTA essentially smooths the decision boundary for morphologically borderline cases.

## 6. Conclusion
*   Summarize the contribution: A mathematically grounded, hierarchical deep learning pipeline that solves extreme class imbalance in carbonate petrography using focused gradient updates (Focal Loss) and topological priors (Hierarchy).
