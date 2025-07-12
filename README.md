
# Machine Learning-Based Intrusion Detection System (ML-IDS)

This project implements a Machine Learning-based Intrusion Detection System (IDS) to enhance network security in enterprise environments. The system uses multiple supervised learning algorithms and is trained on the **TII-SSRC-23** dataset. The goal is to accurately detect malicious activities, including DoS and botnet attacks, with minimal false positives.

---

## 📑 Overview

Traditional IDS often fail to detect zero-day and sophisticated attacks due to their static and signature-based nature. This project leverages supervised machine learning models — such as Random Forest, Gradient Boosting, XGBoost, SVM, KNN, and Logistic Regression — to develop a dynamic and adaptive IDS capable of identifying both known and unknown threats.

---

## 📊 Dataset

- **Name:** TII-SSRC-23 (Secure Systems Research Centre, 2023)
- **Type:** Labeled network flow dataset
- **Contains:** Benign and malicious traffic patterns across various protocols
- **Key Features Used:**
  - Flow Duration
  - Total Forward Packets
  - Total Backward Packets

---

## ⚙️ Implementation Steps

1. **Data Preprocessing:**
   - Handling missing values
   - Feature scaling using `StandardScaler`
   - Categorical encoding (e.g., `LabelEncoder` for protocol types)
   - Correlation-based feature selection

2. **Model Training:**
   - Algorithms tested: Random Forest, Gradient Boosting, XGBoost, SVM, KNN, Logistic Regression
   - Models trained and validated using an 80/20 train-test split
   - GridSearchCV used for hyperparameter tuning

3. **Evaluation Metrics:**
   - Accuracy
   - F1 Score
   - Precision
   - Recall
   - ROC AUC Score

4. **Visualization:**
   - Feature importance plots
   - Confusion matrices
   - Model performance bar charts

---

## 🧠 Best Performing Models

| Model             | Accuracy | F1 Score | ROC AUC |
|------------------|----------|----------|----------|
| Gradient Boosting (Tuned) | 98.15%   | 95.59%   | 97.80%   |
| Random Forest (Tuned)     | 98.10%   | 95.56%   | **99.70%**   |

- **Gradient Boosting**: Slightly better overall classification
- **Random Forest**: Better in ROC AUC, useful for threshold-based alerts

---

## 📌 Key Findings

- **Total Forward Packets** was the most critical feature across models
- Ensemble models outperform individual classifiers in both robustness and accuracy
- ML-based IDS can significantly reduce false positives and improve detection of novel threats

---

## 🛠 Tools & Libraries

- Python (Jupyter Notebook)
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Kaggle cloud environment (for computation)

---

## 🚀 Future Work

- Integration with encrypted traffic analysis
- Application of RNNs and Transformers for improved time-series classification
- Deployment in high-speed real-time enterprise networks
- Adversarial robustness testing

---

## 📄 License

This project is intended for academic and research purposes only.
