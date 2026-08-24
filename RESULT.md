# Diabetes Classifier Training Results

## Training process

The Pima Indians Diabetes dataset contains 768 records, eight input features, and one binary outcome. The data was divided into a stratified 80% training set (614 records) and 20% test set (154 records) using `random_state=42`.

Zero values in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI were treated as missing. Median imputation and standard scaling were fitted only on the training data to prevent data leakage. Five classifiers were then trained and compared: custom KNN, Decision Tree, calibrated SVM, Gaussian Naive Bayes, and Multi-Layer Perceptron (MLP).

## Test results

| Model | Accuracy | Precision | Sensitivity | Specificity | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Custom KNN | 72.73% | 62.50% | 55.56% | 82.00% | 58.82% | 79.00% |
| Decision Tree | **78.57%** | **69.81%** | **68.52%** | **84.00%** | **69.16%** | 78.87% |
| SVM | 74.03% | 65.22% | 55.56% | **84.00%** | 60.00% | 79.63% |
| Gaussian Naive Bayes | 70.13% | 56.67% | 62.96% | 74.00% | 59.65% | 76.46% |
| MLP | 72.73% | 61.54% | 59.26% | 80.00% | 60.38% | **81.04%** |

## Outcome

The Decision Tree achieved the highest test accuracy at 78.57%, while the MLP achieved the highest ROC-AUC at 81.04%. The Streamlit application uses the MLP together with its fitted imputation and scaling steps in one saved pipeline. This application is an educational demonstration and should not be treated as a medical diagnosis.
