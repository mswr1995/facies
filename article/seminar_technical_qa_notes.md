# Technical Q&A Notes

Use these notes after the talk or during questions. The goal is to answer clearly without overclaiming.

---

## 1. Why did you use a hierarchical model instead of a flat classifier?

**Short answer**

Because the grain classes have a natural sedimentological hierarchy, and the rare broken-ooid decision should not be trained directly against thousands of peloids.

**Expanded answer**

A flat classifier treats all four classes as unrelated categories. But geologically, the decision is more structured:

1. peloid or non-peloid
2. ooid-like or intraclast
3. intact ooid or broken ooid

The hierarchy follows this reasoning. It also isolates later decisions from irrelevant majority-class gradients. For example, the broken-ooid head is trained only on ooid-like grains, not on every peloid in the dataset.

**Careful wording**

The hierarchy does not automatically guarantee better performance on every split. In this test split, the flat baseline had higher Balanced Accuracy, but the hierarchical model is more interpretable and had stronger intraclast recall.

---

## 2. Why did the flat ResNet baseline perform better in Balanced Accuracy?

**Short answer**

The flat baseline looked better mainly because it classified all 6 broken ooids correctly. With only 6 test examples, one prediction changes the recall by 16.7 percentage points.

**Expanded answer**

The flat ResNet-18 achieved 86.9% Balanced Accuracy, while the full hierarchical model achieved 83.5%. But the difference is driven by one class with only 6 test samples. The flat model got 6/6 broken ooids; the hierarchical model got 5/6.

That difference is statistically unstable. The hierarchical model performed better on intraclast recall: 71.4% compared with 61.9% for the flat baseline.

**Careful wording**

I would not claim that the hierarchical model universally outperforms the flat baseline. The stronger claim is that the hierarchy provides a geologically interpretable structure and improves rare-class recovery compared with the unmodified hierarchical base.

---

## 3. Why use Balanced Accuracy instead of Overall Accuracy?

**Short answer**

Because the dataset is extremely imbalanced. Overall Accuracy can look high even when rare classes are missed.

**Expanded answer**

Peloids are 87.1% of the dataset. A model that predicts peloid most of the time can achieve high Overall Accuracy without learning the minority classes.

Balanced Accuracy averages recall across classes, so each class contributes equally. This better matches the scientific goal, because rare components such as broken ooids and intraclasts can be geologically important.

**Formula**

Balanced Accuracy is the mean of per-class recall:

```text
BA = (Recall_peloid + Recall_ooid + Recall_broken_ooid + Recall_intraclast) / 4
```

---

## 4. Why is class imbalance such a serious issue here?

**Short answer**

Because the rare classes are not just statistically rare; they are also geologically meaningful.

**Expanded answer**

The dataset contains 2,300 peloids but only 29 broken ooids. That is about a 79:1 ratio.

During training, the model receives many more gradient updates from peloids than from broken ooids. Without intervention, it can learn a decision boundary that favours the majority class.

Oversampling helps because it increases how often the model sees broken ooid examples during training.

---

## 5. Why did oversampling help so much?

**Short answer**

Because with fewer than 20 broken ooids in the training split, the model otherwise sees too little rare-class signal.

**Expanded answer**

The base hierarchical model recovered only 1/6 broken ooids. After targeted oversampling, it recovered 4/6. In the full model, it recovered 5/6.

Oversampling does not create new geological examples. It repeats existing broken ooid examples with augmentation, so the model receives more gradient updates from that class.

**Careful wording**

Oversampling is helpful, but it also risks overfitting because the original number of broken ooids is small. That is why independent validation on new carbonate successions is needed.

---

## 6. What is Focal Loss, in simple terms?

**Short answer**

Focal Loss reduces the influence of easy examples and gives more importance to difficult examples.

**Expanded answer**

In imbalanced datasets, the model can become very good at the majority class and still keep receiving many easy majority-class training examples. Focal Loss down-weights those easy examples, so difficult or minority examples contribute more strongly to learning.

In this study, Focal Loss was applied separately at each decision stage, because each stage has a different local imbalance.

---

## 7. What is staged training?

**Short answer**

Staged training controls which parts of the model learn at different times, so the majority-class decision does not dominate the shared backbone too early.

**Expanded answer**

The model has a shared ResNet-18 backbone and three classification heads. If everything is trained jointly from the beginning, the first decision, peloid vs non-peloid, can dominate learning because peloids are so common.

The staged schedule first trains the heads, then adapts the backbone, then focuses on the minority-class heads, and finally fine-tunes the full model.

**Simple phrasing**

The aim is to prevent the model from learning mostly “peloid features” before it has learned the finer distinctions among the rare classes.

---

## 8. What is test-time augmentation / orientation averaging?

**Short answer**

The same grain is passed through the model in several orientations, and the predictions are averaged.

**Expanded answer**

Carbonate grains do not have a natural upright direction. A broken ooid is still a broken ooid after rotation or flipping.

So at inference time, the model evaluates the original image, flips, and rotations. The predictions are averaged to make the final decision more stable.

**Careful wording**

This helps reduce orientation sensitivity, but it does not solve the underlying data scarcity problem.

---

## 9. Why use ResNet-18?

**Short answer**

ResNet-18 is compact, well-established, and suitable for small datasets compared with larger models.

**Expanded answer**

The dataset has only 2,642 grains. Larger models such as Vision Transformers or deeper ResNets could overfit more easily without substantially more labelled data.

ResNet-18 provides a good balance: enough capacity to learn visual texture, but not so large that it becomes unnecessarily data-hungry.

It was initialised with ImageNet weights, mainly to reuse low-level visual features like edges and textures.

---

## 10. Why not use a Vision Transformer or larger model?

**Short answer**

Because the dataset is small and highly imbalanced. A larger model may overfit without more data.

**Expanded answer**

Transformers can perform very well when enough data is available, but they usually require larger datasets or stronger pretraining. Here, the limiting factor is not just model capacity; it is the small number of minority-class examples.

Future work could test larger models, but only with stronger validation and preferably more annotated carbonates from different settings.

---

## 11. Why did you first try SAM and XGBoost?

**Short answer**

It was a useful pilot pipeline for separating segmentation and classification.

**Expanded answer**

The early approach used SAM to propose grain masks, then extracted hand-crafted geometric, colour, and texture features, and classified them with XGBoost.

This was useful because it tested whether classical feature engineering could solve the problem.

But the subtle differences between classes, especially under diagenesis, were not fully captured by those hand-crafted descriptors. That motivated the shift to end-to-end learning from masked grain patches.

**Careful wording**

SAM + XGBoost was not useless. It helped reveal the limitations of generic segmentation plus hand-crafted features.

---

## 12. Is the model doing segmentation or classification?

**Short answer**

The final model is a classifier, not a segmentation model.

**Expanded answer**

The grains are already manually annotated with polygons. Those polygons are used to extract masked grain patches. The model then classifies each masked patch into one of four classes.

So the current work focuses on grain-level classification, not automatic grain boundary detection.

**If asked about future work**

A complete automated pipeline would need both reliable grain segmentation and classification. This study focuses on the classification stage.

---

## 13. Why are ostracods, bivalves, and quartz excluded?

**Short answer**

Their sample counts were too small for reliable classification.

**Expanded answer**

The excluded classes had very few examples:

- ostracods: 32
- bivalve fragments: 4
- quartz grains: 9

With such low counts, performance estimates would be unreliable and training would be unstable. The study therefore focuses on four classes with enough examples to support a meaningful experiment.

---

## 14. Are 29 broken ooids enough?

**Short answer**

Enough for a proof of concept, but not enough for a definitive general model.

**Expanded answer**

There are only 29 broken ooids in total and only 6 in the test set. This means the result must be interpreted cautiously.

The full model identified 5/6 broken ooids, but the confidence interval is wide. One mistake changes the percentage substantially.

This is why the study is framed as a proof of concept and why independent validation is necessary.

---

## 15. How was the train/validation/test split done?

**Short answer**

The dataset was split into training, validation, and test sets using stratified random sampling.

**Expanded answer**

The split was 60% training, 20% validation, and 20% test, while preserving class proportions.

The test set contained 529 grains:

- 460 peloids
- 42 ooids
- 21 intraclasts
- 6 broken ooids

The test set was held out during model development.

**Careful wording**

Because grains from the same micrograph can share imaging conditions and local diagenetic context, this is within-dataset validation, not full external validation.

---

## 16. Is there data leakage because grains come from the same micrographs?

**Short answer**

There is no pixel-level overlap between grain patches, but grains from the same micrograph may share context.

**Expanded answer**

Each grain patch is masked using its own polygon, so adjacent grains do not share the same grain pixels. However, grains from the same micrograph can share imaging conditions, staining, preparation artefacts, and local geological context.

Therefore, the result should be interpreted as within-dataset performance. External validation on independent thin sections or successions is needed to test generalisation.

---

## 17. What is the main limitation of the study?

**Short answer**

Generalisation is the main limitation.

**Expanded answer**

The dataset comes from one carbonate succession, with 18 micrographs and 2,642 grains. The model has not yet been tested on independent carbonate formations, different labs, or different imaging systems.

The second limitation is the very small number of broken ooids.

So the current result demonstrates feasibility, not universal deployment readiness.

---

## 18. What does the confusion matrix tell us?

**Short answer**

The main remaining error is peloid-intraclast confusion.

**Expanded answer**

The model correctly identifies most peloids and ooids, and 5 out of 6 broken ooids.

The largest error is that some peloids are predicted as intraclasts. This is geologically understandable because advanced micritisation can make peloids and intraclasts both appear dark, rounded, and structureless.

This supports the idea that the model’s errors are interpretable and connected to real petrographic ambiguity.

---

## 19. What is the geological meaning of broken ooids?

**Short answer**

Broken ooids can indicate mechanical reworking under higher-energy conditions.

**Expanded answer**

Ooids form with cortical lamination. When they are mechanically fractured, this can reflect agitation, transport, storm events, wave action, or strong currents.

So identifying broken ooids can help sedimentologists interpret depositional energy and reworking history.

---

## 20. Is the model replacing sedimentologists?

**Short answer**

No. It should be viewed as a decision-support tool.

**Expanded answer**

The model can help quantify grain populations and highlight candidate grains, especially in large datasets. But geological interpretation still requires expert judgement, especially for ambiguous grains and broader facies interpretation.

The aim is to support reproducibility and efficiency, not remove expert interpretation.

---

## 21. What would you do next?

**Short answer**

Test generalisation and expand the dataset.

**Expanded answer**

The next steps would be:

1. Test the model on independent carbonate successions.
2. Add more annotated examples, especially broken ooids and intraclasts.
3. Compare annotation-level splitting with image-level or thin-section-level validation.
4. Improve or automate the segmentation stage.
5. Explore geologically constrained synthetic augmentation for rare classes.

---

## 22. What is the single most important result?

**Short answer**

Targeted oversampling changed broken-ooid recovery from 1/6 to 4/6, and the full model reached 5/6.

**Expanded answer**

The main contribution is not just the final accuracy. It is showing through ablation that rare-grain recovery depends strongly on imbalance-aware training.

The hierarchy gives the model a geological structure, but the rare class still needs enough training signal.

---

## 23. What should I say if someone challenges the small test set?

**Answer**

That is a valid limitation. The broken-ooid test set has only six examples, so the exact percentage should not be overinterpreted. I present it as evidence from this dataset, not as a final generalisation claim.

The stronger result is the ablation pattern: without oversampling, the hierarchical model found only one broken ooid; with targeted oversampling and the full pipeline, it found five. That shows the importance of imbalance-aware training in this setting.

---

## 24. What should I say if someone asks why the flat model is not enough?

**Answer**

The flat model is a valid baseline, and in this split it achieved higher Balanced Accuracy. But its advantage comes from one additional broken-ooid prediction among only six test examples.

The hierarchical model is more interpretable and better aligned with the geological decision process. It also improves intraclast recall compared with the flat baseline.

So I would not say the flat model is useless. I would say the hierarchy provides a geologically meaningful structure that is worth further testing on larger and independent datasets.

---

## 25. Very Short Answers To Common Questions

**Why deep learning?**  
Because the diagnostic features are visual and subtle, and hand-crafted descriptors were limited.

**Why hierarchy?**  
Because the classes follow a natural sedimentological decision tree.

**Why oversampling?**  
Because broken ooids are too rare to provide enough training signal naturally.

**Why Balanced Accuracy?**  
Because Overall Accuracy is misleading when 87% of grains are peloids.

**Main result?**  
Full model recovered 5/6 broken ooids and achieved 83.5% Balanced Accuracy.

**Main limitation?**  
Small dataset from one carbonate succession; generalisation is untested.

**Main geological error?**  
Peloid-intraclast confusion due to micritisation and visual ambiguity.

**Future work?**  
External validation, more rare-class examples, and improved segmentation.

