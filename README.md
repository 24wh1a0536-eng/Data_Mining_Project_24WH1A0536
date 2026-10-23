# Machine Learning-Based Classification of Diabetes Using Medical Diagnostic Data

**Course Activity:** Data Mining (2nd Mid-Term Assignment)  
**Student ID / Register No.:** `24WH1A0536`  
**Dataset Name:** Diabetes Dataset  
**Dataset File:** `diabetes.csv` / `diabetes.arff`

---

## 1. Project Overview

Diabetes is a chronic health condition that affects how the body regulates blood glucose levels. Analyzing medical measurements can help identify patterns associated with recorded diabetes test outcomes.

The goal of this data mining project is to apply supervised machine learning classification techniques to medical diagnostic data and classify observations into two categories: `tested_positive` and `tested_negative`.

The project uses WEKA (Waikato Environment for Knowledge Analysis) to examine the dataset, apply classification algorithms, and evaluate their performance.

**Important:** This project is an academic classification exercise. Its predictions are not a substitute for professional medical diagnosis.

---

## 2. Dataset Hygiene and Preparation

The project uses the Diabetes Dataset provided in CSV and/or ARFF format. The original source file should be preserved, and preprocessing should be performed on a separate working copy.

### 2.1 Dataset Structure

The dataset contains **768 instances and 9 attributes**, consisting of eight input attributes and one target attribute.

The input attributes describe patient-related measurements, while the target attribute records the diabetes test outcome.

### 2.2 Data Quality Considerations

Before applying classification algorithms, the dataset should be inspected for:

- Missing values and invalid measurements.
- Zero values that may represent unavailable measurements for attributes such as plasma glucose, blood pressure, skinfold thickness, insulin, and BMI.
- Unusual values and differences in attribute scales.
- Correct data types and target class labels.

Zero values should not automatically be treated as missing for every attribute. Their interpretation depends on the attribute's meaning.

### 2.3 Target Label Handling

The target attribute is `class`, with two categories:

- `tested_negative`
- `tested_positive`

Instances should retain their original class labels. Target labels must not be invented or inferred during preprocessing.

---

## 3. Class Distribution and Imbalance

The dataset has the following class distribution:

| Class | Instance Count | Class Percentage |
|---|---:|---:|
| `tested_negative` | 500 | 65.10% |
| `tested_positive` | 268 | 34.90% |
| **Total** | **768** | **100.00%** |

The dataset has an unequal distribution between the two classes.

### Data Mining Implications for WEKA

- A ZeroR classifier that always predicts `tested_negative` would achieve approximately **65.10% accuracy**, assuming the stated class counts.
- Accuracy alone may not adequately describe model performance.
- Precision, recall, F1-score, ROC area, the confusion matrix, and other suitable metrics should be considered.
- Stratified cross-validation can help evaluate the model while preserving the approximate class distribution across folds.
- Resampling or cost-sensitive classification may be explored if justified by the evaluation results.

---

## 4. Prevention of Target Leakage

Target leakage occurs when information that directly reveals the target class is included among the predictor attributes.

For this project:

1. **Target Attribute (`class`):** The class label must be used as the prediction target, not as an input feature.
2. **Derived Information:** Any feature created using the target label or information that directly reveals it must be excluded from the predictors.
3. **Preprocessing:** Operations such as imputation, feature selection, and resampling should be fitted using training data only when evaluating a model, to avoid information leaking from the test data.
4. **Original Attributes:** The eight medical input attributes should be examined for validity and suitability before classification.

No additional target-leakage columns should be claimed to exist unless they are confirmed in the actual dataset.

---

## 5. Predictor Feature Definitions and Biological Meaning

The dataset contains eight input attributes and one target attribute.

| Attribute Name | Expected Type | Definition |
|---|---|---|
| `preg` | Numeric | Number of pregnancies |
| `plas` | Numeric | Plasma glucose concentration |
| `pres` | Numeric | Diastolic blood pressure |
| `skin` | Numeric | Triceps skinfold thickness |
| `insu` | Numeric | Two-hour serum insulin level |
| `mass` | Numeric | Body mass index (BMI) |
| `pedi` | Numeric | Diabetes pedigree function, representing a family-history-related measure |
| `age` | Numeric | Age of the individual |
| `class` | Nominal (Target) | Recorded outcome: `tested_negative` or `tested_positive` |

**Note:** The attribute names and meanings should be checked against the supplied CSV or ARFF header before final submission. Missing-value counts and observed ranges should be calculated from the actual file rather than assumed.

---

## 6. How to Load and Validate the Dataset

### Prerequisites

- WEKA 3.8 or a compatible version.
- The supplied `diabetes.arff` or `diabetes.csv` file.
- Python with `pandas` if additional data inspection is required.

### Step 1: Open the Dataset in WEKA

1. Launch the WEKA GUI Chooser.
2. Select **Explorer**.
3. Open the **Preprocess** tab.
4. Click **Open file...** and select `diabetes.arff`.
5. Verify the number of instances and attributes.
6. Select `class` as the class attribute.

If only the CSV file is available, open it in WEKA and verify that the attributes and class labels have been interpreted correctly. Alternatively, convert it to ARFF format.

### Step 2: Inspect the Dataset

Check the following:

1. Total number of instances and attributes.
2. Attribute types and observed ranges.
3. Missing values and suspicious zero measurements.
4. Distribution of `tested_negative` and `tested_positive`.
5. Whether `class` is correctly selected as the target attribute.

### Step 3: Apply Classification Algorithms

Suitable algorithms to explore in WEKA include:

- J48 Decision Tree
- Naive Bayes
- Random Forest
- Logistic Regression
- ZeroR as a baseline classifier

Use a consistent evaluation method, such as stratified 10-fold cross-validation, to compare the selected algorithms fairly.

### Step 4: Evaluate Model Performance

Record the following measures from WEKA's evaluation output:

- Correctly and incorrectly classified instances.
- Overall accuracy.
- Precision and recall for each class.
- F-measure for each class.
- Confusion matrix.
- ROC area, where appropriate.

The best-performing model should be selected based on multiple relevant evaluation measures, rather than accuracy alone.

---

## 7. Scope and Objectives

The primary objectives of this project are:

1. To understand the structure of the Diabetes Dataset.
2. To examine the input attributes and target class.
3. To identify missing, invalid, or unusual measurements.
4. To perform appropriate preprocessing where necessary.
5. To apply supervised classification algorithms using WEKA.
6. To compare the classification performance of different algorithms.
7. To interpret the results using suitable evaluation metrics.

---

## 8. Conclusion

This project demonstrates the application of data mining techniques to medical diagnostic data. By examining patient-related attributes and applying supervised classification algorithms, the study aims to evaluate how effectively different models classify recorded diabetes test outcomes.

The project provides practical experience in data preprocessing, classification, cross-validation, and model evaluation. The reliability of its conclusions depends on the quality of the supplied data, appropriate experimental design, and careful interpretation of the results.