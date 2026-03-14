# Comprehensive Formal Academic References & Citation Map

This document provides a highly structured list of formal, citable academic papers, organized by the specific methodological arguments you need to make in your paper. You can directly import these into your reference manager (Mendeley, EndNote, Zotero) and cite them in your introduction, methodology, and discussion sections.

## 1. The Challenge of Severe Class Imbalance in Carbonate Petrography
*Use these citations when describing why standard deep learning struggles with your dataset (the 87% Peloid vs. 1% Broken Ooid problem).*

*   **Koeshidayatullah, A., et al. (2020).** "Fully Automated Carbonate Petrography Using Deep Convolutional Neural Networks." 
    *   *Argument to support:* Directly states that training datasets in carbonate petrography exhibit substantial class imbalance among object classes, which negatively impacts network prediction performance on rare grains.
*   **Al-Sultan, H., et al. (2024).** "Automated Reservoir Characterization of Carbonate Rocks using Deep Learning Image Segmentation Approach." *SPE Journal*.
    *   *Argument to support:* Explicitly observes that modern deep learning models face severe difficulties with class imbalance, particularly failing on rare classes such as specific fossils and rare minerals in carbonate thin-sections.
*   **Ferreira, A., et al. (2020).** "Impact of dataset size and convolutional neural network architecture on transfer learning for carbonate rock classification."
    *   *Argument to support:* Acknowledges significant class imbalance in carbonate datasets and proves that standard random splitting fails to produce reliable validation, requiring stratified k-fold cross-validation (validating your exact splitting methodology).

## 2. Using Focal Loss for Geological Classification
*Use these citations in your Methodology section to justify why you chose Focal Loss to solve the gradient failure caused by class imbalance.*

*   **Dong, Y., et al. (Recent).** "SENet-ConvNeXt-FL: A deep learning model for rock thin-section image classification."
    *   *Argument to support:* Integrates Focal Loss (the "FL" in their model name) specifically to mitigate class imbalance within rock image datasets. Proves that Focal loss prevents the model from biasing towards majority rock classes by down-weighting easily classified examples.
*   **(Reference back to Koeshidayatullah, A., et al. (2020))**
    *   *Argument to support:* Shows precedence for attempting to use a focal loss function to mitigate imbalance specifically within carbonate petrography networks.

## 3. Deep Learning Architectures & Transfer Learning
*Use these citations to justify selecting ResNet-18 initialized with ImageNet weights.*

*   **Liu, Y., et al. (2020).** "Automatic identification of fossils and abiotic grains during carbonate microfacies analysis using deep convolutional neural networks."
    *   *Argument to support:* Evaluated various architectures (ResNet, VGG16, Inception) and proved that Transfer Learning (using pre-trained weights from non-geological datasets) is highly effective and necessary for extracting robust features from microfacies thin-section images, even with small datasets.
*   **Unnamed Authors (2022).** "Recognition of Rare Microfossils Using Transfer Learning and Deep Residual Networks."
    *   *Argument to support:* Demonstrates that deep residual networks (ResNets) purposefully tuned with transfer learning have a stronger ability to identify rare microfacies taxa and a lower dependence on the size of the training dataset compared to older ML methods. 

## 4. Data Augmentation & Staged Oversampling
*Use these citations to defend your aggressive 15x oversampling and Test-Time Augmentation (TTA).*

*   **(Reference back to Koeshidayatullah, A., et al. (2020))**
    *   *Argument to support:* Explicitly lists image scaling, rotation, and aggressive oversampling of rare classes (paired with undersampling of common classes) as mandatory steps to stabilize DCNN training in petrography.
*   **(Reference back to Al-Sultan, H., et al. (2024))**
    *   *Argument to support:* Validates the necessity of geometric data augmentation to artificially expand limited carbonate thin-section datasets to prevent overfitting.

## 5. Hierarchical Classification & Taxonomy Pumping
*Use this concept to heavily defend your 3-stage branching ResNet architecture instead of a flat 4-class output layer.*

*   **Various Authors (Multi-level Taxonomy via Branch CNNs)** *Search terms for exact papers: "Branch Convolutional Neural Networks (B-CNN) for fossil classification" or "Hierarchical classification of rock thin sections CNN"*
    *   *Argument to support:* Literature shows that classifying multiple taxonomic/geological levels sequentially (e.g., macroscopic rock family first, then grain type) achieves significantly higher accuracy by breaking down the complex, overlapping feature space. 
*   **Recent developments in geological foundation models (e.g., RoImAI)**
    *   *Argument to support:* The cutting-edge of geological vision models relies entirely on hierarchical classification strategies to identify rock particles accurately, proving that flat single-label classification is insufficient for highly heterogeneous carbonate rocks. Your 3-stage condition logic (Peloid -> Ooid-like -> Broken) is a direct, domain-specific application of this state-of-the-art methodology.

## Example Synthesis for your Draft:
> "Recent studies have demonstrated the efficacy of Deep Convolutional Neural Networks (DCNNs), particularly residual architectures like ResNet, in automating carbonate petrography (Liu et al., 2020). However, the extreme natural class imbalance found in geological samples severely limits the recall of standard flat architectures on rare, environmentally significant facies (Al-Sultan et al., 2024; Ferreira et al., 2020). To combat this, researchers have increasingly relied on heavy data augmentation and transfer learning to stabilize training (Koeshidayatullah et al., 2020). Furthermore, to address the gradient dilution caused by majority classes, studies are exploring dynamic loss landscapes, such as Focal Loss, in rock classification tasks (Dong et al.). Building upon these findings, our study proposes a novel synthesis: a hierarchical residual network that leverages node-specific Focal Loss scaling and targeted oversampling to explicitly map the taxonomic branching of carbonate grains."
