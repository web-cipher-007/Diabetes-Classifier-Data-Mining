# Diabetes Classifier

An educational machine-learning project that compares five classifiers on the [Pima Indians Diabetes Database hosted by UCI Machine Learning on Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). A Streamlit app uses the trained MLP pipeline to estimate diabetes risk from eight clinical measurements.

The saved pipeline applies the same training-only median imputation and standard scaling used during evaluation, preventing target leakage and training–inference mismatch.

> This project is for educational purposes and is not a medical diagnostic tool.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (managed automatically by uv when needed)

## Run the app

Install the locked dependencies:

```bash
uv sync
```

Start Streamlit:

```bash
uv run streamlit run src/streamlit_app.py
```

Open <http://localhost:8501>, enter the eight requested measurements, and select **Predict Risk**. A value of `0` represents missing or unknown data only for glucose, blood pressure, skin thickness, insulin, and BMI.

The repository already includes the trained artifact at `out/models/diabetes_pipeline.joblib`, so retraining is not required before running the app.

## Retrain the model

Open the notebook:

```bash
uv run jupyter lab src/diabetes_model_training.ipynb
```

Run all cells in order. The notebook:

1. Loads and inspects the dataset.
2. Creates a stratified 80/20 train-test split.
3. Learns imputation and scaling from the training set only.
4. Trains and compares KNN, Decision Tree, SVM, Gaussian Naive Bayes, and MLP models.
5. Saves the fitted MLP pipeline to `out/models/diabetes_pipeline.joblib`.

## Results

The Decision Tree achieved the highest test accuracy at **78.57%**, while the deployed MLP achieved the highest ROC-AUC at **81.04%**. See [RESULT.md](RESULT.md) for the complete comparison.

## Build the report

The reproducible LaTeX report is in `src-tex`. With Tectonic and Poppler installed, rebuild its figures, PDF, and local citation/consistency/similarity audit with:

```bash
make -C src-tex
```

See [src-tex/README.md](src-tex/README.md) for report-specific notes and limitations.

## Project structure

```text
dataset/pima_indians_diabetes.csv   Dataset
src/diabetes_model_training.ipynb  Training and evaluation notebook
src/model_inference.py             Prediction helpers
src/project_paths.py               Shared project paths
src/streamlit_app.py               Streamlit application
out/models/diabetes_pipeline.joblib Trained deployment pipeline
RESULT.md                           Training summary and metrics
src-tex/report.tex                  LaTeX report source
src-tex/report.pdf                  Rendered report
```
