# ML Assignment 2
## Problem Statement
The goal of this project is to build and compare multiple supervised machine learning classification models to predict whether a bank client will subscribe to a term deposit (`y`). The project covers the complete machine learning workflow, including data preprocessing, model training, evaluation using multiple metrics, and deployment of an interactive Streamlit application to demonstrate model performance on uploaded test data.

---

## Dataset Description
- **Dataset Name:** Bank Marketing
- **Type:** Binary Classification (`y`: yes/no)
- **Records:** 45,211 instances (public dataset)
- **Features:** 16 input features (mix of numerical and categorical)
- **Target Column:** `y` (mapped to 1 for "yes" and 0 for "no")
- **Source:** UCI Machine Learning Repository
- **Source Link :** https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Models Used
The following models were trained on the same training dataset and evaluated on the same test dataset:
1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## Model Comparison Table (Evaluation Metrics)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8986 | 0.8500 | 0.6257 | 0.3318 | 0.4336 | 0.4070 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| **Random Forest** | 0.8587 | 0.9223 | 0.4444 | **0.8318** | **0.5793** | **0.5394** |
| XGBoost | **0.9079** | **0.9323** | 0.6591 | 0.4405 | 0.5280 | 0.4912 |

---

## Observations (Model-wise)
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong baseline with good accuracy and AUC, but relatively low recall (0.3478), indicating conservative prediction of positive subscriptions. |
| Decision Tree | Achieves balanced precision and recall, but lower AUC (0.7015) suggests weaker probability ranking and possible overfitting. |
| KNN | Shows accuracy comparable to Logistic Regression, but low recall due to class imbalance and high-dimensional feature space after encoding. |
| Naive Bayes | Captures more positive cases with higher recall (0.5198), but low precision leads to more false positives, reflecting its strong independence assumptions. |
| **Random Forest** | Achieves the **best F1 score (0.5793)** and **highest MCC (0.5394)** with very high recall (0.8318), making it highly effective at identifying positive subscribers in an imbalanced dataset. |
| XGBoost | Delivers the highest AUC (0.9323) and strong overall accuracy, offering a balanced trade-off between precision and recall, though slightly lower F1/MCC than Random Forest. |

---

## Repository Structure
- `model/` contains separate Python files for each model and a training orchestration script.
- `model/artifacts/` contains saved trained model pipelines used for Streamlit inference.
- `data/test.csv` is generated from the dataset split and is used for Streamlit uploads.
- `datapreperation/` contains configuration, preprocessing, and metric utility modules.
- `sample/` contains a sample test CSV file to help users understand the expected input format.

---

## Streamlit Application
### Features Implemented (as per assignment):
- Upload test CSV file
- Select model from dropdown
- Display evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Display confusion matrix and classification report

---

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Train All Models :
   ```bash
   python -m model.train_all

3. Run Stream Lit App 
   ```
   streamlit run app.py
