# Credit Risk Default Risk using Machine Learning models: Accuracy 97%

Credit risk analysis is a crucial task for financial institutions as it enables them to determine the likelihood of default for potential borrowers. In this project, we analyze credit risk using logistic regression and other machine learning models on the American Express dataset. Our aim is to identify the best performing model in predicting credit card defaults and to determine the most important variables in credit risk analysis. Our study shows that XGBoost is the best performing model, with an accuracy of 0.97, precision of 0.91, F1-score of 0.91, AUC value of 0.92. Logistic Regression and other models also performed well, but not as well as XGBoost. Our findings indicate that the most significant variables in predicting credit card defaults are credit score, credit limit utilization, and number of days employed. Furthermore, we find that the age of the borrower is not a significant factor in predicting credit card defaults. This highlights the importance of considering other variables when analyzing credit risk. Our study provides practical implications for financial institutions in improving their credit risk analysis models. By using machine learning techniques such as XGBoost, they can better identify and manage credit risk, thus reducing their losses due to defaults.

## Dataset

The data used in this project comes from the **"AmExpert 2021 CODELAB - Machine Learning Hackathon"** competition hosted on the online coding platform, HackerEarth. The dataset can be accessed here, belongs to American Express, a company that provides customers with various payment products and services.

The original dataset consisted of 45528 rows and 19 columns, but for this study, a subset of 30000 rows and 19 columns was used. The target variable of our data frame is “credit_card_default”, which is a binary variable, whose values are 0 and 1. Credit card default risk is the chance that companies or individuals will not be able to return the money lent on time. Data frame has 6 categorical features and 13 numeric features.

Dataset can be download at: https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021

# Credit Risk Analysis

This project is a credit risk analysis system that predicts the likelihood of loan default using advanced machine learning techniques. By analyzing various customer and financial data points, the project helps financial institutions make informed decisions to minimize risk.
Performance: Achieved high accuracy and interpretability in predicting credit risk.

## Experimental result
Model         | Accuracy
------------- | -------------
Logistic Regression | 0.9464
Random Forest | 0.9650
Decision Tree | 0.9663
LightGBM  | 0.9668
KNN | 0.9681
CatBoost | 0.9683
XGBoost | 0.9734

## Exploratory Data Analysis (EDA)
EDA focuses on understanding the dataset and identifying trends:

- **Distribution Analysis:** Visualizing key variables like loan amounts and credit scores.
- **Correlation Matrix:** Identifying relationships between features.
- **Class Imbalance:** Examining the distribution of high-risk and low-risk labels.

## Preprocessing and Feature Engineering
Key steps in preprocessing include:

1. **Missing Value Treatment:** Filling or imputing missing data points.
2. **Feature Encoding:** Converting categorical variables to numerical values using Label Encoding and One-Hot Encoding.
3. **Feature Scaling:** Standardizing numerical features using MinMaxScaler for model efficiency.
4. **Data Splitting:** Dividing the dataset into training and testing sets (80:20 split).

## Model Training
Multiple machine learning models were implemented and evaluated, including:

- **Logistic Regression:** For baseline classification.
- **Random Forest:** For robust ensemble learning.
- **Gradient Boosting:** For improved predictive accuracy.

### Model Evaluation
The models were assessed using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

## Results
- **Performance:** The Gradient Boosting model achieved the highest accuracy on the test set.
- **Insights:** Identified key factors influencing credit risk, such as credit history and debt-to-income ratio.

Example:

- Input: Customer with a credit score of 700 and a debt-to-income ratio of 30%.
- Prediction: Low Risk.

## How to Run the Code

1. Install Dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```

2. Clone the Repository:
   ```bash
   git clone <repository-link>
   cd <repository-directory>
   ```

3. Prepare the Dataset:
   - Ensure the dataset is in the project directory.
   - Update the file path in the notebook.

4. Run the Notebook:
   Open and execute the notebook `Credit_Risk_Analysis.ipynb` step by step.

5. Test the Model:
   Modify the test inputs in the notebook to predict credit risk for new customers.

## Future Enhancements
- **Deep Learning Models:** Implement neural networks for enhanced predictions.
- **Real-time Predictions:** Deploy the model as a web app using Flask or Streamlit.
- **Additional Features:** Incorporate external datasets to improve model robustness.
- **Explainable AI:** Use SHAP or LIME to explain model predictions for better transparency.

---

