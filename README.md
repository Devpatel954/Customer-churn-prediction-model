# 🏦 Customer Churn Prediction Model

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-lightgrey)

## 📋 Project Overview

This project focuses on predicting customer churn using an **Artificial Neural Network (ANN)**. The goal is to help banks identify customers who are likely to leave and take proactive steps to retain them. By analyzing customer data, we can build a model that predicts whether a customer will churn based on various features like demographics, account details, and behavior patterns.

---

## 🛠️ Tech Stack & Tools

- **Languages**: Python
- **Libraries**: TensorFlow, Keras, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, VS Code
- **Techniques**: Data Preprocessing, Feature Engineering, ANN Model Building, Hyperparameter Tuning, Model Evaluation

---

## 🚀 Project Features

- **Data Preprocessing**: Handled missing values, encoded categorical features, and scaled numerical data for optimal model performance.
- **Feature Engineering**: Selected key features to improve the model's predictive power.
- **Model Building**: Constructed a deep learning model using TensorFlow and Keras with a focus on binary classification.
- **Model Optimization**: Tuned hyperparameters (layers, neurons, activation functions) to enhance model accuracy.
- **Evaluation Metrics**: Assessed model performance using accuracy, precision, recall, and F1-score.

---

## 📊 Data

The dataset used includes information on customers such as:

- **Demographics**: Age, Gender, Geography
- **Account Information**: Credit Score, Balance, Tenure, Number of Products, Estimated Salary
- **Behavioral Data**: Active Status, Has Credit Card, Is a Member

The dataset is sourced from [Kaggle](https://www.kaggle.com).

---

## 🧠 Model Architecture

The Artificial Neural Network (ANN) was built using the **TensorFlow Keras API**:

- **Input Layer**: 11 features
- **Hidden Layers**: 
  - **1st Hidden Layer**: 64 neurons, ReLU activation
  - **2nd Hidden Layer**: 32 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation for binary classification

### **Training & Optimization**
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1 Score

The architecture was designed to efficiently handle the binary classification task of predicting whether a customer will churn or not.

---

## 📈 Results

The model's performance was evaluated using multiple metrics:

- **Accuracy**: 85%
- **Precision**: 82%
- **Recall**: 78%
- **F1 Score**: 80%

These results indicate that the model can effectively predict customer churn, providing a good balance between precision and recall. This allows the bank to identify at-risk customers more accurately and take proactive measures to retain them.

---

## 📝 Insights & Future Work

### 🔍 Key Insights
- Customers with **low credit scores** and **higher account balances** are more likely to churn.
- Demographic factors like **geography** and **gender** significantly impact churn rates.
- Customers with **fewer products** and **low engagement** (inactive accounts) are at higher risk of leaving.

### 🚀 Future Enhancements
- **Deployment**: Deploy the model as a web application using Flask or Django for real-time predictions.
- **Data Enrichment**: Integrate additional customer data such as transaction history and customer service interactions to improve model accuracy.
- **Model Improvement**: Experiment with ensemble methods (e.g., Random Forest, Gradient Boosting) to boost performance.
- **Explainability**: Implement model interpretability techniques (like SHAP values) to understand feature importance better.

---

## ⭐ Acknowledgements

Special thanks to [Kaggle](https://www.kaggle.com) for providing the dataset and to the **open-source community** for their invaluable resources. 
