# Diabetes Prediction Using Data Mining Algorithms in Weka

**Course Activity:** Data Mining / Weka Tool Activity  
**Academic Year:** 2026–2027  
**Year & Sem:** 3rd Year / 1st Sem  

---

### Student & Institutional Details

- **Student Name:** D. Tulasi Priya
- **Roll Number / Register ID:** 24WH1A0536
- **Department:** Department of Computer Science and Engineering
- **Institution / College:** BVRIT HYDERABAD College of Engineering for Women
- **Laboratory / Subject:** Data Mining Laboratory (Weka Practical Activity)

---

## 1. Abstract

This project explores the use of data mining classification algorithms to predict diabetes test outcomes using the Pima Indians Diabetes dataset. The dataset contains 768 instances and 9 attributes: eight predictors and one target attribute, `class`, with values `tested_negative` and `tested_positive`.

Three classifiers—**J48**, **Naive Bayes**, and **RandomForest**—were evaluated in Weka using stratified 10-fold cross-validation. Naive Bayes achieved the highest accuracy at **76.3021%**, followed by RandomForest at **75.2604%** and J48 at **73.8281%**. Naive Bayes also had the highest positive-class recall of **61.2%**. However, it still missed a substantial number of positive cases. The results show why accuracy should be considered alongside precision, recall, F-measure, ROC area, and confusion matrices.

This is an educational data-mining activity, not a medical diagnostic system.

---

## 2. Problem Statement

### 2.1 Formal Problem Statement

> **To compare machine-learning classification algorithms for predicting the diabetes test outcome from the attributes in the Pima Indians Diabetes dataset using Weka.**

### 2.2 Objectives

1. Load and inspect the diabetes dataset in Weka.
2. Understand the dataset attributes and class labels.
3. Check the dataset structure and preprocessing status.
4. Apply J48, Naive Bayes, and RandomForest.
5. Evaluate the models using stratified 10-fold cross-validation.
6. Compare accuracy, class-specific metrics, and confusion matrices.
7. Identify the best-performing algorithm among the three tested models.

### 2.3 Importance of Category-Specific Evaluation

Accuracy alone does not show how well a model identifies each class. The negative class has more instances than the positive class, so precision, recall, F-measure, and the confusion matrix should also be considered. Recall for `tested_positive` indicates how many actual positive cases were correctly identified.

### 2.4 Educational Scope Disclaimer

> **Important Disclaimer:** This project is strictly an educational data-mining activity. It is not a certified clinical diagnostic system and must not be used for medical diagnosis, treatment planning, or patient management.

---

## 3. Dataset Description

### 3.1 Dataset Overview

- **Input File:** `diabetes.arff`
- **Dataset Type:** Tabular classification dataset
- **Total Instances:** 768
- **Total Attributes:** 9 (8 predictors and 1 class attribute)
- **Target Attribute:** `class`
- **Class Values:** `tested_negative`, `tested_positive`
- **Evaluation Method:** Stratified 10-fold cross-validation

Each instance represents a record with medical measurements and demographic information. The class attribute records the diabetes test outcome.

### 3.2 Target Class and Predictor Attributes

| # | Attribute Name | Typical Meaning | Role |
| :-: | :--- | :--- | :--- |
| 1 | `preg` | Number of times pregnant | Predictor |
| 2 | `plas` | Plasma glucose concentration | Predictor |
| 3 | `pres` | Diastolic blood pressure | Predictor |
| 4 | `skin` | Triceps skin-fold thickness | Predictor |
| 5 | `insu` | 2-hour serum insulin | Predictor |
| 6 | `mass` | Body mass index (BMI) | Predictor |
| 7 | `pedi` | Diabetes pedigree function | Predictor |
| 8 | `age` | Age | Predictor |
| 9 | `class` | `tested_negative` or `tested_positive` | **Target Class** |

*Verify the exact attribute names and order against the supplied ARFF file before submitting.*

### 3.3 Class Distribution

The Weka confusion matrices indicate these class totals:

- **`tested_negative`:** 500 instances
- **`tested_positive`:** 268 instances
- **Total:** 768 instances

The classes are not perfectly balanced. Both overall accuracy and positive-class recall should therefore be considered.

---

## 4. Data Preprocessing

The ARFF dataset was loaded into Weka and inspected before classification. This report describes preprocessing as a **validation and inspection step**; it does not claim that the original values were changed or that missing values were imputed.

### 4.1 Preprocessing Checks

1. Open `diabetes.arff` in Weka Explorer using the **Preprocess** tab.
2. Check that the dataset contains 768 instances and 9 attributes.
3. Inspect attribute types and confirm `class` is the nominal target.
4. Review missing-value counts and attribute summaries.
5. Inspect histograms and value distributions for unusual values.
6. Select `class` as the target before running the classifiers.
7. Keep the original ARFF unchanged unless a separately documented preprocessing operation is applied.

### 4.2 Preprocessing Considerations

Some measurements in diabetes datasets may contain zero values that are medically unusual for certain attributes. A zero should not automatically be replaced or treated as missing without a documented rule. If Normalize or Standardize filters were applied, record the exact filter and settings shown in Weka.

---

## 5. Methodology

The workflow follows these phases:

```text
[Load ARFF Dataset]
        |
        v
[Inspect Attributes and Values]
        |
        v
[Check Missing / Suspicious Values]
        |
        v
[Select class Attribute]
        |
        v
[Configure J48, Naive Bayes, RandomForest]
        |
        v
[Run Stratified 10-Fold Cross-Validation]
        |
        v
[Record Accuracy and Class Metrics]
        |
        v
[Compare Confusion Matrices]
        |
        v
[Discuss Results and Limitations]
```

### Workflow Steps

1. Load `diabetes.arff` in Weka Explorer.
2. Inspect the instance count, attribute count, data types, and class values.
3. Review attribute summaries and missing-value information.
4. Select `class` as the target attribute.
5. Choose J48, Naive Bayes, and RandomForest from the Classify tab.
6. Evaluate each classifier using stratified 10-fold cross-validation.
7. Record correctly and incorrectly classified instances.
8. Compare precision, recall, F-measure, ROC area, and confusion matrices.
9. Identify the strongest result and its limitations.

---

## 6. Experimental Setup

- **Mining Software:** Weka Explorer
- **Input File:** `diabetes.arff`
- **Instance Count:** 768
- **Attribute Count:** 9
- **Target Attribute:** `class`
- **Class Labels:** `tested_negative`, `tested_positive`
- **Validation Scheme:** Stratified 10-fold cross-validation
- **Classifiers:** J48, Naive Bayes, RandomForest

*For strict reproducibility, copy the Weka version, classifier options, relation name, and random seed from the actual experiment output.*

---

## 7. Algorithm Descriptions

### 7.1 J48 (Decision Tree)

J48 is Weka's implementation of the C4.5 decision-tree algorithm. It builds a tree by selecting attributes that help separate the classes. Internal nodes test attributes, branches represent test outcomes, and leaves provide predictions.

**Advantage:** The tree can be inspected to understand the model's decision rules.

### 7.2 Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It estimates the probability that an instance belongs to each class, assuming attributes are conditionally independent given the class.

**Advantage:** It is computationally efficient and useful as a classification baseline.

### 7.3 RandomForest

RandomForest combines multiple decision trees trained with randomized samples and feature selection. The final class prediction is obtained by voting across the trees.

**Advantage:** Combining trees can improve generalization compared with relying on a single decision tree.

### 7.4 Rationale for Comparing the Algorithms

J48 learns interpretable decision rules, Naive Bayes estimates class probabilities, and RandomForest combines many trees. Comparing these approaches shows how different classifiers perform on the same dataset.

---

## 8. Results and Comparison

### 8.1 Overall Performance Summary

The following results were recorded from Weka using 10-fold cross-validation.

| Classifier | Correctly Classified | Incorrectly Classified | Accuracy |
| :--- | :---: | :---: | :---: |
| **J48** | 567 / 768 | 201 | **73.8281%** |
| **Naive Bayes** | 586 / 768 | 182 | **76.3021%** |
| **RandomForest** | 578 / 768 | 190 | **75.2604%** |

**Best accuracy:** Naive Bayes, at **76.3021%**.

### 8.2 J48 (Decision Tree)

- **Correctly Classified:** 567
- **Incorrectly Classified:** 201
- **Accuracy:** 73.8281%

| Class | True Positive Rate (Recall) | False Positive Rate | Precision | F-Measure | ROC Area | PRC Area |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tested_negative` | 0.814 | 0.403 | 0.790 | 0.802 | 0.751 | 0.811 |
| `tested_positive` | 0.597 | 0.186 | 0.632 | 0.614 | 0.751 | 0.572 |

**J48 Confusion Matrix**

| Actual / Predicted | `tested_negative` | `tested_positive` |
| :--- | :---: | :---: |
| **Actual `tested_negative`** | **407** | 93 |
| **Actual `tested_positive`** | 108 | **160** |

J48 correctly classified 407 negative cases and 160 positive cases. It classified 108 actual positive cases as negative.

### 8.3 Naive Bayes

- **Correctly Classified:** 586
- **Incorrectly Classified:** 182
- **Accuracy:** 76.3021%
- **Weighted Precision:** 0.759
- **Weighted Recall:** 0.763
- **Weighted F-Measure:** 0.760

| Class | True Positive Rate (Recall) | False Positive Rate | Precision | F-Measure | ROC Area | PRC Area |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tested_negative` | 0.844 | 0.388 | 0.802 | 0.823 | 0.818 | 0.891 |
| `tested_positive` | 0.612 | 0.156 | 0.678 | 0.643 | 0.818 | 0.670 |

**Naive Bayes Confusion Matrix**

| Actual / Predicted | `tested_negative` | `tested_positive` |
| :--- | :---: | :---: |
| **Actual `tested_negative`** | **422** | 78 |
| **Actual `tested_positive`** | 104 | **164** |

Naive Bayes achieved the highest accuracy. It correctly classified 164 of 268 positive cases, while 104 actual positive cases were classified as negative.

### 8.4 RandomForest

- **Correctly Classified:** 578
- **Incorrectly Classified:** 190
- **Accuracy:** 75.2604%

| Class | True Positive Rate (Recall) | False Positive Rate | Precision | F-Measure | ROC Area | PRC Area |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `tested_negative` | 0.834 | 0.399 | 0.796 | 0.814 | 0.816 | 0.889 |
| `tested_positive` | 0.601 | 0.166 | 0.660 | 0.629 | 0.816 | 0.681 |

**RandomForest Confusion Matrix**

| Actual / Predicted | `tested_negative` | `tested_positive` |
| :--- | :---: | :---: |
| **Actual `tested_negative`** | **417** | 83 |
| **Actual `tested_positive`** | 107 | **161** |

RandomForest correctly classified 161 positive cases and 417 negative cases. Its accuracy was below Naive Bayes but above J48.

---

## 9. Interpretation and Discussion

### 9.1 Comparative Performance

1. **Naive Bayes achieved the highest accuracy:** 586 of 768 instances were classified correctly, resulting in 76.3021% accuracy and a weighted F-measure of 0.760.
2. **RandomForest ranked second:** It achieved 75.2604% accuracy and a positive-class ROC area of 0.816.
3. **J48 ranked third by accuracy:** It achieved 73.8281% accuracy and offers an inspectable decision tree.
4. **The accuracy gap is modest:** Naive Bayes is about 2.47 percentage points above J48. These results do not establish that one model will generalize best to every population.

### 9.2 Positive-Class Performance

Recall for `tested_positive` was:

- **J48:** 59.7%
- **Naive Bayes:** 61.2%
- **RandomForest:** 60.1%

Naive Bayes had the highest positive-class recall among the tested models, but it still missed 104 of the 268 actual positive cases in the reported cross-validation results. This is why overall accuracy should not be used alone.

### 9.3 Confusion Matrix Interpretation

A confusion matrix compares actual classes with predicted classes. Diagonal cells show correct predictions, while off-diagonal cells show misclassifications. For Naive Bayes, 422 negative cases and 164 positive cases were correctly classified. However, 104 actual positive cases were predicted as negative. In a medical context, such false negatives are a key reason why this classroom model must not be treated as a diagnostic system.

---

## 10. Limitations

1. **Classification errors:** All three models misclassified some records.
2. **Positive cases were missed:** Positive-class recall ranged from 59.7% to 61.2%.
3. **Dataset representativeness:** Results from this dataset do not guarantee the same performance on other populations.
4. **Preprocessing transparency:** Normalization, standardization, missing-value handling, and any special handling of zero values should be documented exactly as performed.
5. **Validation scope:** Stratified 10-fold cross-validation does not replace external validation on independent data.
6. **No clinical validity:** This activity does not establish clinical safety or suitability for patient management.

---

## 11. Conclusion and Future Work

### 11.1 Conclusion

This data-mining activity compared J48, Naive Bayes, and RandomForest on the Pima Indians Diabetes dataset using Weka. The dataset contained 768 instances and 9 attributes, including the target attribute `class`.

**Naive Bayes achieved the highest accuracy of 76.3021%**, followed by RandomForest at 75.2604% and J48 at 73.8281%. Naive Bayes also achieved a positive-class recall of 61.2%, but it still missed a substantial number of positive cases. The results demonstrate the importance of considering precision, recall, F-measure, and confusion matrices alongside accuracy.

The project provides practical experience in loading an ARFF dataset, inspecting attributes, running classification algorithms, and interpreting Weka outputs. The results are for academic learning only and must not be used to diagnose diabetes.

### 11.2 Future Work

1. **Document preprocessing:** Record the exact filters, missing-value handling, and any treatment of zero values.
2. **Explore additional classifiers:** Compare other Weka classifiers under the same validation method.
3. **Study class-sensitive evaluation:** Focus on recall, precision, ROC area, and PRC area for `tested_positive`.
4. **Validate on independent data:** Test whether results generalize to an external dataset.
5. **Improve reproducibility:** Record the Weka version, classifier options, class selection, and random seed.

---

## 12. References

1. **WEKA.** *The Waikato Environment for Knowledge Analysis.* Available: `https://www.cs.waikato.ac.nz/ml/weka/`.
2. **Frank, E., Hall, M. A., & Witten, I. H.** (2016). *The WEKA Workbench: Online Appendix for Data Mining: Practical Machine Learning Tools and Techniques* (4th ed.). Morgan Kaufmann.
3. **Quinlan, J. R.** (1993). *C4.5: Programs for Machine Learning.* Morgan Kaufmann.
4. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
5. **John, G. H., & Langley, P.** (1995). Estimating Continuous Distributions in Bayesian Classifiers. *Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence*, 338–345.
6. **Pima Indians Diabetes dataset.** Add the exact source URL or citation for the copy used in this project.

---

## Appendix: Submitted Project Artifacts

Include the artifacts that are actually present in your project folder:

1. **Dataset**
   - `diabetes.arff` — ARFF dataset loaded into Weka.
   - `diabetes.csv` — CSV version, if used.
2. **Weka Results**
   - Preprocess tab screenshot showing the relation, instance count, attributes, and class distribution.
   - J48 classifier output screenshot.
   - Naive Bayes classifier output screenshot.
   - RandomForest classifier output screenshot.
3. **Preprocessing and Validation**
   - `validate_preprocessing.py` — validation script, if used.
   - `diabetes_preprocessed.csv` — exported dataset, if generated.
4. **Documentation**
   - `Weka_Tool_Activity_Report.md` — editable Markdown report.
   - `Diabetes_Data_Mining_Report.tex` — LaTeX version, if used.
   - `README.md` — project overview, if included.

*Before submission, replace the student placeholders, verify the exact dataset source, and ensure all reported values and screenshots match your own Weka output.*
