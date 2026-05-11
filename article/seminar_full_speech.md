# Full Seminar Speech

**Presentation title:** Hierarchical grain facies classification in carbonate thin sections using deep learning

**Target length:** about 12 minutes  
**Style:** simple, clear, meaningful  

---

## Slide 1: Title

Good [morning/afternoon]. My name is Musawer Muradi.

Today I will present my work on hierarchical grain facies classification in carbonate thin sections using deep learning.

The main idea of this work is simple: carbonate grain classification should not be treated only as a generic computer vision problem. The model should also reflect how sedimentologists actually reason about these grains.

---

## Slide 2: Agenda

Here is the structure of the talk.

I will start with the geological motivation and thin-section petrography. Then I will explain the carbonate grain components and why manual classification is difficult.

After that, I will describe the dataset and the class imbalance problem. Then I will explain why a general computer vision approach is not enough, and how the hierarchical deep learning method was designed.

Finally, I will show the evaluation results and the main takeaways.

---

## Slide 3: Why carbonate grain classification matters

Carbonate grains are important because they record information about the environment where the rock was deposited.

Their shape, texture, and internal fabric can tell us about depositional energy, reworking, diagenesis, and palaeoenvironmental change.

For example, broken ooids and intraclasts can be especially meaningful. They may indicate high-energy conditions, such as storm activity, wave reworking, or strong currents.

In the image, we can see a thin-section view where these grains preserve geological information at a microscopic scale.

So the goal is not only to classify images. The goal is to recover geological signals from thin-section imagery.

---

## Slide 4: Thin-section petrography

Thin-section petrography is one of the standard methods in sedimentary geology.

Rock samples are sliced and ground to around 30 micrometers thickness, so that they can be observed under a polarising microscope.

In this study, the images were taken under plain polarised light.

The visual information in these images includes grain boundaries, internal textures, and signs of diagenetic alteration.

On this slide, the images show that thin sections contain many grains with different shapes and internal fabrics. These are the features that a sedimentologist uses during interpretation, and they are also the features that the model must learn from.

---

## Slide 5: Carbonate grain facies components

This study focuses on four carbonate grain types.

The first one is peloid. Peloids are usually micritic and often appear dark and structureless.

The second one is ooid. Ooids are usually recognised by their concentric cortical lamination.

The third one is broken ooid. These are ooids that have been mechanically fractured, so only part of the cortex may be visible.

The fourth one is intraclast. Intraclasts are reworked carbonate grains, and they may show angular boundaries or remnant internal fabric.

Some other components were present in the dataset, such as ostracods, bivalve fragments, and quartz grains, but their numbers were too small for reliable modelling. Therefore, they were excluded from the classification experiments.

The important point here is that these four classes are not equally common, and they are not always visually easy to separate.

---

## Slide 6: Why this is a hard, manual problem

Manual interpretation is difficult for several reasons.

First, it requires expert petrographic knowledge. These are not simple visual categories.

Second, diagenesis can erase or weaken the diagnostic textures. For example, an ooid can lose much of its lamination, and then it may start to look similar to a peloid.

Third, peloids, altered ooids, and intraclasts may overlap visually. In some cases, even human interpretation becomes uncertain.

The image on this slide shows examples of this ambiguity. Some grains have very weak internal structure, and the boundary between classes is not always sharp.

So the model is not just dealing with image noise. It is dealing with real geological ambiguity.

---

## Slide 7: Dataset

The dataset contains 2,642 annotated carbonate grains.

These grains come from 18 plain polarised light micrographs and 9 thin sections.

Each grain was manually annotated using polygon boundaries. After annotation, each grain was extracted as a 96 by 96 pixel masked patch.

This means the model does not classify the whole thin-section image. It classifies individual grains.

The pipeline shown here summarises the process: we start from a thin-section micrograph, then expert polygon annotations are used to extract masked grain patches. These patches are passed to the hierarchical ResNet model, and the output is the predicted grain facies.

This grain-level setup is important because sedimentological interpretation depends on the individual components inside the rock.

---

## Slide 8: Extreme class imbalance

The core machine learning problem is class imbalance.

The dataset is strongly dominated by peloids. There are 2,300 peloids, but only 29 broken ooids.

Ooids have 210 examples, and intraclasts have 103 examples.

This means the ratio between peloids and broken ooids is about 79 to 1.

The graph makes this imbalance very clear. The peloid bar dominates the dataset, while the broken ooid class is very small.

This is a serious problem because a model can get high overall accuracy simply by predicting the majority class most of the time.

But scientifically, that is not enough. The rare classes can be the most meaningful ones.

---

## Slide 9: Why Balanced Accuracy matters

Because of this imbalance, overall accuracy can be misleading.

If a model predicts peloid for almost everything, it can still look good numerically, because peloids make up 87.1 percent of the labelled grains.

Broken ooids are only 1.1 percent of the dataset.

So in this work, Balanced Accuracy is used as the main metric.

Balanced Accuracy gives equal importance to each class by averaging the recall of all classes.

The graph on the right also shows the test set distribution. There are 460 peloids in the test set, but only 6 broken ooids.

This means that broken-ooid percentages must be interpreted carefully. One error changes the recall by 16.7 percentage points.

---

## Slide 10: The classes are visually close

Another difficulty is visual similarity.

Peloids are often micritic and structureless.

Ooids depend on visible cortical lamination, but this lamination can be weakened by diagenesis.

Broken ooids depend on detecting a fractured cortex, and sometimes this fractured part is only a small part of the grain.

Intraclasts may be recognised by angularity or remnant fabric, but these features can also be subtle.

The image examples show that these classes are not visually obvious object categories. This is not like classifying cars, animals, or buildings.

These are fine petrographic distinctions, and some of them are difficult even for human interpretation.

---

## Slide 11: Why general computer vision is not enough

A general computer vision model usually works best when there are many examples and when the visual categories are clearly separable.

Carbonate grain classification violates both assumptions.

There are not many examples of the rare classes, and the classes can overlap in shape, colour, and texture.

In the early stage of the work, a more conventional pipeline was tested. It used the Segment Anything Model for segmentation, then hand-crafted features, and then XGBoost for classification.

This was useful as a pilot study, because it showed what the problem looks like computationally.

But generic segmentation and hand-crafted geometric or texture features were not enough to fully capture the subtle petrographic differences.

So the method was changed toward a model that learns directly from masked grain patches, while also following geological reasoning.

---

## Slide 12: A hierarchy that follows sedimentological reasoning

The main idea is to replace one flat four-class decision with a hierarchy of simpler geological questions.

The first question is: is this grain a peloid?

If it is not a peloid, the second question is: is it ooid-like, or is it an intraclast?

If it is ooid-like, the third question is: is it intact, or is it broken?

The diagram shows this decision process.

This hierarchy is useful for two reasons.

First, it is closer to how sedimentologists think about these grains.

Second, it helps isolate the rare broken-ooid decision from the peloid-dominated training signal.

Instead of forcing one flat classifier to separate all classes at once, the model solves the problem step by step.

---

## Slide 13: Hierarchical ResNet-18 and training strategy

The model uses a ResNet-18 backbone to extract visual features from each masked grain patch.

On top of this shared backbone, there are three binary classification heads, corresponding to the three stages of the hierarchy.

The first head separates peloid from non-peloid.

The second head separates ooid-like grains from intraclasts.

The third head separates intact ooids from broken ooids.

The training strategy is also designed for imbalance.

Focal Loss is used to reduce the effect of easy majority-class examples and focus more on difficult cases.

Broken ooids are oversampled, because there are very few of them.

Staged training is used to reduce early dominance by the first decision, which is the peloid decision.

Finally, orientation averaging is used at inference time, because grains do not have a fixed upright direction.

The key point is that the architecture alone is not enough. Rare-class recovery also requires imbalance-aware training.

---

## Slide 14: What improved rare-grain recovery?

This is the most important result slide.

The hierarchical base model, without targeted oversampling, recovered only 1 out of 6 broken ooids in the test set.

After adding targeted oversampling, the model recovered 4 out of 6 broken ooids.

In the full model, with staged training and orientation averaging, it recovered 5 out of 6 broken ooids.

Balanced Accuracy improved from 63.3 percent to 83.5 percent.

The graph shows that the largest improvement comes from oversampling.

This means that the rare-class problem is not solved only by using a better architecture. The model also needs to see the rare examples often enough during training.

For this dataset, targeted oversampling was the critical step for broken-ooid recovery.

---

## Slide 15: Full model performance and remaining errors

On the held-out test set of 529 grains, the full model achieved 83.5 percent Balanced Accuracy and 92.1 percent Overall Accuracy.

Broken ooid recall was 83.3 percent, which means 5 out of 6 broken ooids were correctly identified.

Intraclast recall was 71.4 percent.

The confusion matrix shows where the model still makes errors.

The main remaining confusion is between peloids and intraclasts. In particular, some peloids are predicted as intraclasts.

This makes geological sense. Under advanced micritisation, both peloids and intraclasts can appear dark, rounded, and structureless.

So the model’s main error is not random. It reflects a real petrographic boundary that can also be difficult for human analysts.

This is important because it means the model can be interpreted as a decision-support tool, not as a black box that simply produces labels.

---

## Slide 16: Limitations and take-home message

There are also important limitations.

First, the dataset comes from one carbonate succession.

Second, there are only 29 broken ooids in total.

Third, generalisation to other formations, laboratories, and imaging conditions remains untested.

So this work should be understood as a proof of concept, not as a universal carbonate classifier.

The take-home message is that rare grains are important but underrepresented.

Geological hierarchy helps structure the learning problem.

And imbalance-aware training is essential for recovering rare classes.

In short, petrographic AI should encode geological structure, not only generic computer vision machinery.

---

## Slide 17: Thank you

Thank you for listening.

I would be happy to take your questions.

