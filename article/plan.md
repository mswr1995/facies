# Article Revision Plan
## Target: Not necessarily Computers & Geosciences — Major Revision → Acceptance

---

## ⚠️ BLOCKED ITEMS — Resolve Before Finalising Article

These items cannot be edited until the model is re-run or verified. Everything else in this plan can proceed in parallel.

---

### B1. CRITICAL — Balanced Accuracy vs Overall Accuracy mislabel `[NEEDS MODEL RUN]`

**What was found:**
The `Balanced Acc.` column in Table 1 (ablation) is actually **Overall Accuracy**, not Balanced Accuracy.

Proof from the confusion matrix the paper already contains:
- Peloid recall: 451/460 = 98.0%
- Ooid recall: 39/42 = 92.9%
- Broken recall: 5/6 = 83.3%
- Intraclast recall: 14/21 = 66.7%
- **True Balanced Accuracy = (98.0 + 92.9 + 83.3 + 66.7) / 4 = 85.2%**

The paper claims **96.2%** — but 96.2% = 509/529, which is overall accuracy.
Every row in Table 1 "Balanced Acc." column matches overall accuracy, not the mean of per-class recalls.

**Why it matters:**
The XGBoost baseline (80.9%) IS the true balanced accuracy. The DL numbers are overall accuracy. You are comparing two different metrics in the same table and calling both "Balanced Accuracy."

**What to do when re-running:**
1. Add a correct `balanced_accuracy_score` print in `scripts/evaluate_final.py` using `sklearn.metrics.balanced_accuracy_score`
2. Run it on the `exp9_tta_results.json` checkpoint
3. If true balanced accuracy = 85.2%: update article headline to 85.2% (or report both metrics — see decision below)
4. If code produces a different number: there is a bug — find it and fix it

**Decision to make after re-run:**
- Option A: Report both metrics (96.2% overall + 85.2% balanced). Story: flat DL collapses to 61% balanced acc; our model holds at 85%.
- Option B: Report only balanced accuracy (85.2%). Cleaner but less dramatic headline.
- Option C: Report only overall accuracy (96.2%) and reframe XGBoost comparison using overall accuracy too.

**Affects:** Abstract, Section 6.1, Table 1, Table 2, Conclusion.

---

### B2. Flat ResNet-18 Balanced Accuracy number is missing `[NEEDS MODEL RUN]`

**What was found:**
The paper says the hierarchical model "substantially improved" over a flat ResNet-18 (Section 6.1) but gives no number. This is a gap reviewers will flag.

**What we have from existing results:**
- `exp2_data_augmentation`: flat model with strong augmentation → **61.1% true balanced accuracy** on test set
- `exp3_class_weighting`: flat model with class weights → **81.2% true balanced accuracy** on test set
- Neither is labelled "flat ResNet-18 standard cross-entropy baseline"

**What to do when re-running:**
1. Identify which experiment is the intended "flat ResNet-18 + standard cross-entropy" baseline
2. If it is exp2 (augmented flat): balanced accuracy = 61.1%, overall = 92.8%
3. If it was never run separately, run `train.py` without the hierarchical model and evaluate
4. Report the number in Section 6.1 and the abstract

**Affects:** Abstract (item 3.2), Section 6.1 (item 7.2).

---

### B3. Group K-Fold vs grain-level split — need to verify what was actually used `[NEEDS CODE AUDIT]`

**What was found:**
`scripts/create_grain_split.py` uses a simple `train_test_split` at the grain level **without image grouping**. This means grains from the same image can appear in both train and test — exactly the leakage the article claims to prevent.

The article claims "Strict Stratified Group K-Fold split (where the 'Group' explicitly refers to the source image)."

**What to check:**
1. Open `data/processed/train_split.json` and `data/processed/test_split.json`
2. Check whether any `image_name` values appear in both files
3. If yes → the split does NOT prevent image-level leakage, and the claim in Section 5.1 is wrong
4. If no → the split happens to be clean by chance (or a different split script was used)

Run this to check:
```bash
python3 -c "
import json
train = json.load(open('data/processed/train_split.json'))
test = json.load(open('data/processed/test_split.json'))
train_imgs = set(g['image_name'] for g in train['grains'])
test_imgs = set(g['image_name'] for g in test['grains'])
print('Overlap:', train_imgs & test_imgs)
print('Train images:', sorted(train_imgs))
print('Test images:', sorted(test_imgs))
"
```

**If overlap exists:** The methodological claim must be corrected (and results may be artificially inflated).
**If no overlap:** The split is image-clean even if the code doesn't enforce it explicitly. Update Section 5.1 to accurately describe what was done.

**Affects:** Section 5.1 (item 7.1), and potentially all results if leakage is found.

---

### B4. XGBoost per-class recall on the 529-grain test set `[NEEDS VERIFICATION]`

**What was found:**
Table 1 shows dashes for XGBoost per-class recall. The `findings.md` has XGBoost per-class numbers but they may be from a different test set (the faciesnet XGBoost project used its own split).

Numbers from `findings.md` for reference:
- Peloid recall: 77.4%
- Ooid recall: 87.5%
- Intraclast recall: 87.5%
- Broken ooid: F1 = 66.7% (recall unknown precisely)

**What to do:**
1. Check whether the XGBoost from `faciesnet` was evaluated on the same 529-grain test set used by the deep learning model
2. If yes: add these numbers to Table 1
3. If no: either re-run XGBoost on the DL test set, or remove XGBoost from Table 1 entirely and reference it in prose only

**Affects:** Table 1 (item 8.1).

---

### SUMMARY: Status of all items

| Status | Items |
|--------|-------|
| **[DONE] — already applied to article** | 1.4, 1.5, 2.2, 3.1, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 10.1, 11.1, 11.2, 12.1, 12.2 (partial) |
| **Waiting for Gallois** | 1.1, 1.2, + expand 9.2 geological interpretation |
| **Waiting for you** | 2.1 (grain patch images) |
| **Wait for model re-run** | B1 (metric label), B2 (flat baseline number → 3.2, 7.2) |
| **Wait for code audit** | B3 (split verification → 7.1) |
| **Wait for XGBoost check** | B4 (Table 1 XGBoost row → 8.1) |
| **Bibliography incomplete** | alsultan2024, ferreira2020, liu2020, liudunham2023 still missing volume/pages |

---

## HOW TO USE THIS FILE

Each item has a status tag:
- `[NEEDS YOU]` — requires geological facts or numbers only you/Gallois have
- `[NEEDS NUMBER]` — requires a result from a model run
- `[READY TO FIX]` — I can edit the LaTeX right now, no new information needed
- `[DONE]` — completed and applied to final_article.tex

Add your comments/answers directly under each item. Once you answer a `[NEEDS YOU]` item, I will make the edit.

---

## SECTION 1: GEOLOGICAL SETTING (Highest Risk — Likely Rejection Without This)

### 1.1 Formation name, geographic location, stratigraphic age `[NEEDS YOU]`
The section currently says only "marine carbonate successions" with no specifics.
Required additions:
- Formation name (e.g., "Shuaiba Formation", "Asmari Formation", etc.)
- Country / basin / field name
- Stratigraphic age (e.g., "Upper Jurassic", "Early Cretaceous")
- Depositional environment (e.g., "shallow-water carbonate ramp", "reef margin")

**Your answer:**
> I will ask Arnauld Gallois(other auther) to write it for me - Leave a reminder in the article

**Status:** Reminder flag `[GEOLOGICAL SETTING — TO BE COMPLETED BY GALLOIS]` already inserted into article at Section 3.1.

---

### 1.2 Microscopy specifications `[NEEDS YOU]`
Required additions:
- PPL (plane-polarised light) or XPL (cross-polarised light) or both?
- Magnification used (e.g., 2.5×, 5×, 10×)
- Microscope model or type (transmitted-light petrographic microscope)
- Image resolution / pixel size if known

**Your answer:**
> I will ask Arnauld Gallois(other auther) to write it for me - Leave a reminder in the article

**Status:** Reminder flag `[MICROSCOPY DETAILS — TO BE COMPLETED BY GALLOIS]` already inserted into article at Section 3.2.

---

### 1.3 Annotation protocol `[NEEDS YOU — PARTIALLY DONE]`
Required additions:
- How many annotators? (you, Gallois, both?)
- Was inter-rater agreement checked? If yes, Cohen's κ value?
- What tool was used for annotation? (LabelMe — I can see this from the JSON format)
- Were annotation guidelines written, or was it expert judgment?

**Your answer:**
> Gallois was the annotator, I did all computer related editing/processing, I dont know inter-rater (not-checked). labelme was used then I converted it to more common one, which i dont remember. The guidelines doesnt exist but I converted them to xgboost, cnn, yolo, sam friendly versions

**Status [PARTIALLY DONE]:** Article updated to state Gallois as single annotator using LabelMe (`\citep{labelme}`). Inter-rater agreement omission noted as a limitation. Excluded classes disclosed. Full annotation conversion pipeline not described in article (not required for journal submission).

---

### 1.4 Excluded classes — must be disclosed `[DONE]`
~~The raw data contains 32 Ostracods, 4 Bivalves, 9 Quartz grains. These are excluded from the study but never mentioned.~~

**Applied:** One sentence added to Section 3.2 disclosing the excluded 32 Ostracod, 4 Bivalve, and 9 Quartz grain instances.

---

### 1.5 Class distribution table `[DONE]`
~~I have the exact counts from the raw JSON files. I will create a LaTeX table showing per-class grain counts.~~

**Applied:** Table `tab:class_distribution` added to Section 3.2 showing all per-class counts including excluded classes.

---

## SECTION 2: FIGURES (High Priority — Visual Paper With Zero Figures)

### 2.1 Figure 1: Example grain patches per class `[NEEDS YOU]`
A reviewer cannot evaluate a visual classification paper without seeing what the classes look like.

Needed: one representative grain patch per class (4 images total) arranged as a 2×2 or 1×4 grid with labels.

**Your answer:**
> We will produce them together after we fix other things.

**Status:** Placeholder `\begin{figure}...\end{figure}` block exists in article. Waiting for image paths.

---

### 2.2 Architecture diagram `[DONE]`
~~A schematic of the hierarchical branching (shared backbone → 3 binary heads → final class).~~

**Applied:** Full TikZ architecture figure (`fig:architecture`) added in Section 4.3 showing input → ResNet-18 backbone → MLP(1) [α=0.25] → Peloid / MLP(2) [α=0.50] → Intraclast / MLP(3) [α=0.75] → Ooid / Broken Ooid, with all α values labeled on each head.

---

## SECTION 3: ABSTRACT

### 3.1 "fail categorically" — too absolute `[DONE]`
~~Change to more measured language.~~

**Applied:** Changed to "struggle severely under these conditions, collapsing into majority-class detectors with near-zero recall for rare but geologically significant grain types."

---

### 3.2 Missing flat ResNet-18 balanced accuracy number `[NEEDS NUMBER → see B2]`
The abstract says the model outperforms "a flat deep learning baseline" but never states the number.

**Blocked on B2.** Once flat baseline BA is confirmed, insert the number here.

---

## SECTION 4: INTRODUCTION

### 4.1 Zero citations across three paragraphs `[DONE]`
~~Add appropriate citations to each paragraph.~~

**Applied:** `\citep{koeshidayatullah2020, pireslima2020}` added to paragraph 1; `\citep{ferreira2020}` added to paragraph 2.

---

### 4.2 Contributions buried in prose → numbered list `[DONE]`
~~Will convert contribution description to a numbered enumerate list.~~

**Applied:** Three-item `\begin{enumerate}` contribution list added to Introduction paragraph 3.

---

### 4.3 "sequential noise reduction" is vague `[DONE]`
~~Replace with clearer taxonomic routing explanation.~~

**Applied:** Replaced with explanation of taxonomic routing: each head is exposed only to the sub-population relevant to its binary decision.

---

## SECTION 5: LITERATURE REVIEW

### 5.1 Dalhat (2025) citation mismatch `[DONE]`
~~Dalhat (2025) is about broken rock surfaces and 40 mineral types — not carbonate microfacies.~~

**Applied:** Dalhat moved to a broader geological ML context paragraph; carbonate annotation bottleneck argument now supported by `\citep{alsultan2024, ferreira2020}`.

---

### 5.2 Xie et al. (2025) misuse `[DONE]`
~~Their paper addresses cross-domain lithology, not intra-domain morphological ambiguity.~~

**Applied:** Paragraph reframed to correctly distinguish inter-domain vs. intra-domain imbalance challenges.

---

### 5.3 Dong citation incomplete in .bib `[DONE]`
~~The .bib entry for `dong` had no journal name, no volume, no pages.~~

**Applied:** `dong` key renamed to `dong2025`; full citation added (Dong, Fengjuan et al., *Carbonates and Evaporites*, vol. 40, 2025, doi: 10.1007/s13146-025-01140-x). All in-text `\citep{dong}` updated to `\citep{dong2025}`.

---

### 5.4 New literature: hierarchical classification precedents `[DONE]`
~~HD-CNN and B-CNN should be cited in Section 2.3 or 2.5.~~

**Applied:** `\citep{yan2015hdcnn}` added to Section 2.3 for hierarchical CV precedent. Full BibTeX entry for Yan et al. (ICCV 2015) added to references.bib.

*Note: B-CNN (Zhu et al. 2017, arXiv:1709.09890) was not added — it is a preprint without a peer-reviewed venue. Can be added if you want.*

---

### 5.5 Carbonate petrography textbook reference `[DONE]`
~~Adding Flügel (2010) textbook to legitimize the grain type classification scheme.~~

**Applied:** `\citep{flugel2010}` added to Section 3 (Geological Setting). Full BibTeX entry for Flügel (2010) *Microfacies of Carbonate Rocks*, Springer added to references.bib.

---

## SECTION 6: METHODOLOGY

### 6.1 Binary mask ≠ spatial attention mechanism `[DONE]`
~~Section 4.1 incorrectly called a binary pixel mask a "spatial attention mechanism."~~

**Applied:** Changed to "A binary grain mask is applied to each patch, zeroing all background pixels and isolating the grain's internal morphology and boundary structure from the surrounding cementing matrix."

---

### 6.2 α values need one sentence of justification `[DONE]`
~~Add one sentence explaining why Stage 1 has α=0.25 and Stage 3 has α=0.75.~~

**Applied:** Justification paragraph added: Stage 1 faces 87% Peloids → α=0.25 aggressive down-weighting; Stage 3 faces near-balanced Whole vs Broken Ooid → α=0.75 minimal adjustment.

---

### 6.3 K=6 must be defined in the TTA equation `[DONE]`
~~Confirmed K=6 transforms from exp9_tta.py. Must be stated explicitly in Section 4.4.~~

**Applied:** K=6 defined explicitly with all 6 transforms listed (identity, h-flip, v-flip, 90°, 180°, 270° rotations) in Section 4.4.

---

### 6.4 Oversampling factor 15× not stated in methodology text `[DONE]`
~~The 15× oversampling only appeared in Table 1 caption.~~

**Applied:** "Broken Ooid instances within Stage 3 training batches are oversampled at a 15× rate" added to Section 4.4.

---

## SECTION 7: EXPERIMENTAL SETUP

### 7.1 Group K-Fold vs. 60/20/20 fixed split — needs clarification `[NEEDS CODE AUDIT → see B3]`
Whether the split actually enforces image-level grouping is unverified.

**Blocked on B3.** Run the diagnostic command in B3, then update Section 5.1 based on what is actually true.

---

### 7.2 Flat ResNet-18 Balanced Accuracy `[NEEDS NUMBER → see B2]`
Same as 3.2. The number is needed in Section 6.1.

**Blocked on B2.**

---

## SECTION 8: RESULTS

### 8.1 XGBoost per-class recall — dashes in Table 1 `[NEEDS VERIFICATION → see B4]`
XGBoost per-class numbers in findings.md may be from a different test set.

**Blocked on B4.** Once verified, either fill in per-class recall or remove XGBoost row and describe in prose only.

---

### 8.2 Table 2 mixes three different metrics `[DONE]`
~~The comparison table mixes Accuracy, Top-1 Accuracy, and Balanced Accuracy in the same column.~~

**Applied:** "Metric" column added to Table 2; imbalance ratio corrected from 85:1 to ~79:1; caption improved.

---

### 8.3 "perfectly recall 5 out of 6" — word error `[DONE]`
~~"Perfectly" is factually wrong (83.3% ≠ perfect).~~

**Applied:** Changed to "correctly classify 5 of the 6 extreme minority Broken Ooid instances."

---

### 8.4 Peloid percentage inconsistency throughout `[DONE]`
~~Inconsistently stated as "over 85%" vs "87%" in different places.~~

**Applied:** Standardized to "87%" throughout the article (actual value: 87.1%).

---

### 8.5 85:1 imbalance ratio is wrong `[DONE]`
~~Table 2 caption stated "85:1 imbalance ratio." Actual ratio = 2300:29 = 79.3:1.~~

**Applied:** Corrected to "~79:1" in Table 2 caption.

---

## SECTION 9: DISCUSSION

### 9.1 "dataset noise" for Peloids `[DONE]`
~~Section 7.1: "discarding 87% of the dataset noise." Peloids are a real grain type.~~

**Applied:** Changed to reflect correct routing logic: "correctly identifies and routes 87% of samples — the Peloid majority — to their output node. This taxonomic routing allows MLP(2) and MLP(3) to dedicate their entire parameter space exclusively to the non-Peloid sub-population."

---

### 9.2 No geological interpretation — missing subsection `[DONE]`
~~New Section 7.3 needed answering: Peloid-Intraclast confusion, Broken Ooid high-energy indicator, practical use guidance.~~

**Applied:** New Section 7.3 "Geological Interpretation of Classification Patterns" added with:
- Peloid-Intraclast confusion explained as expected due to morphological similarity
- Broken Ooid detection framed as a high-energy depositional indicator
- Flag for Gallois to expand with formation-specific interpretation

---

### 9.3 Only two Discussion subsections `[DONE]`
~~With Section 7.3 added, discussion now has three subsections.~~

**Applied:** Three subsections present: 7.1 (hierarchical routing), 7.2 (imbalance strategy), 7.3 (geological interpretation).

---

## SECTION 10: LIMITATIONS

### 10.1 Wilson CI ordering — reader loses confidence `[DONE]`
~~CV result should come before Wilson CI, not after.~~

**Applied:** Rewritten to lead with CV result (81% ± 9%) then present Wilson CI [36%–97%] as the caveat for the 6-instance test set.

---

## SECTION 11: CONCLUSION

### 11.1 Rewrite needed — currently repeats introduction `[DONE]`
~~Conclusion should lead with results, not methodology.~~

**Applied:** Conclusion rewritten to: (1) empirical results demonstrated, (2) practical implication for geologists, (3) future work (more formations, more data).

---

### 11.2 "mathematically robust foundation" — vague `[DONE]`
~~This phrase says nothing.~~

**Applied:** Replaced with a concrete statement about demonstrated performance and practical significance.

---

## SECTION 12: BIBLIOGRAPHY

### 12.1 Dong citation — incomplete `[DONE]`
~~The bib entry had no journal, volume, pages, or DOI.~~

**Applied:** See item 5.3 — complete entry now in references.bib as `dong2025`.

---

### 12.2 Most bib entries missing volume/issue/pages `[PARTIALLY DONE]`
**Done:**
- `koeshidayatullah2020` — Frontiers in Earth Science, vol. 8, 2020, DOI added
- `dong2025` — Carbonates and Evaporites, vol. 40, 2025, full entry
- `yan2015hdcnn` — ICCV 2015, pp. 2740–2748, full entry
- `flugel2010` — Springer book, full entry
- `labelme` — Wada (2016) GitHub, full entry

**Still incomplete (TODO comments in .bib):**
- `alsultan2024` — no volume/pages (SPE Journal 2024)
- `ferreira2020` — no volume/pages (Computers & Geosciences 2020)
- `liu2020` — no volume/pages (Sedimentary Geology 2020)
- `liudunham2023` — no volume/pages (Geoenergy Science and Engineering 2023)

---

## PRIORITY ORDER — REMAINING WORK

**Waiting for Gallois (ask him directly):**
1. Items 1.1, 1.2 — formation name, location, age, microscopy specs
2. Item 9.2 expansion — geological interpretation of confusion patterns (Section 7.3 placeholder exists)

**Waiting for you:**
3. Item 2.1 — grain patch images for Figure 1 (one per class, 4 images)

**After model re-run:**
4. B1 → decide which metric to headline (85.2% balanced vs 96.2% overall) → update Abstract, Table 1, Table 2, Conclusion
5. B2 → get flat baseline number → fill into Abstract + Section 6.1

**After code audit:**
6. B3 → run diagnostic, then fix or confirm Section 5.1 split description

**After XGBoost check:**
7. B4 → fill or remove XGBoost per-class recall in Table 1
