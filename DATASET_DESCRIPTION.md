# DATASET DESCRIPTION

## Diabetes Dataset

**Course Activity:** Data Mining – 2nd Mid-Term Academic Assignment  
**Student Register Number:** `24WH1A0536`  
**Dataset Name:**  Diabetes Dataset  
**Dataset Files:** `diabetes.csv`, `diabetes.arff`  
**Total Instances:** 768  
**Total Attributes:** 9  
**Input Attributes:** 8  
**Target Attribute:** `class`

---

## 1. Dataset Background

The Pima Indians Diabetes Dataset is a widely used dataset for studying diabetes classification using data mining and machine learning techniques. It contains medical diagnostic measurements collected from female patients of Pima Indian heritage who were at least 21 years old.

The dataset is commonly used for educational and research purposes to explore relationships between medical measurements and recorded diabetes test outcomes.

The objective of using this dataset is to understand the characteristics of the data and classify observations into positive and negative diabetes test result categories.

## 2. Dataset Structure

The dataset contains 768 instances and 9 attributes. It consists of eight numerical input attributes and one nominal target attribute.

| Attribute | Data Type | Description |
|---|---|---|
| `preg` | Numeric | Number of times the patient has been pregnant. |
| `plas` | Numeric | Plasma glucose concentration measured during an oral glucose tolerance test. |
| `pres` | Numeric | Diastolic blood pressure in mm Hg. |
| `skin` | Numeric | Triceps skinfold thickness in millimetres. |
| `insu` | Numeric | Two-hour serum insulin level in μU/mL. |
| `mass` | Numeric | Body mass index (BMI). |
| `pedi` | Numeric | Diabetes pedigree function, representing information related to family history. |
| `age` | Numeric | Age of the patient in years. |
| `class` | Nominal | Recorded diabetes test outcome. |

## 3. Target Attribute

The target attribute is `class`, which contains two categories:

- `tested_negative`: The recorded diabetes test result is negative.
- `tested_positive`: The recorded diabetes test result is positive.

The target attribute is used to distinguish between the two classes during classification.

## 4. Class Distribution

| Class | Number of Instances | Percentage |
|---|---:|---:|
| `tested_negative` | 500 | 65.10% |
| `tested_positive` | 268 | 34.90% |
| **Total** | **768** | **100%** |

The dataset contains more negative observations than positive observations.

## 5. Dataset Characteristics

- **Numerical attributes:** Eight input attributes containing medical measurements and patient-related information.
- **Categorical attribute:** One target attribute representing the recorded test outcome.
- **Classification type:** Binary classification.
- **Data format:** CSV and ARFF.
- **Software compatibility:** The ARFF format can be loaded directly into WEKA for data mining analysis.

## 6. Limitations

1. The dataset represents a specific population and may not generalize to all populations.
2. Some zero values in medical measurement attributes may represent unavailable measurements.
3. The classes are not equally distributed.
4. The dataset's recorded test outcomes should not be treated as a substitute for professional medical diagnosis.

## 7. Conclusion

The Diabetes Dataset contains 768 instances and nine attributes, including eight numerical predictors and one nominal target attribute. It provides a useful dataset for understanding medical data, examining attribute characteristics, and studying binary classification techniques in data mining.

## References

1. UCI Machine Learning Repository: https://archive.ics.uci.edu/
2. WEKA Data Mining Software: https://www.cs.waikato.ac.nz/ml/weka/
