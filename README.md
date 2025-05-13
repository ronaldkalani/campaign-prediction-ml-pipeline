Predictive Campaign Targeting & Premium Segmentation (ML Pipeline)  

This project builds a complete machine learning pipeline to identify customers most likely to respond to insurance cross-sell offers. Using the Health Insurance Cross-Sell Prediction dataset from Kaggle, it includes data cleaning, feature engineering, model training with XGBoost, predictive evaluation, automated submission generation, and Excel exports for dashboarding. The solution supports marketing campaign optimization and is designed for BI teams, data analysts, and business stakeholders.

Goal: Build a predictive ML pipeline that segments customers using demographic and behavioral features, predicts cross-sell interest, and supports automated campaign execution.

Intended Audience:  

- BI Analysts: monitor and analyze customer segments  
- Marketing Teams: use model predictions for targeting  
- Sales Operations: improve ROI through intelligent targeting  
- Data Scientists: evaluate modeling pipeline  
- Executives: track churn and engagement trends via dashboards

Pipeline Steps (with code):  

1. Data Loading & Exploration  
import pandas as pd  
train = pd.read_csv('train.csv')  
test = pd.read_csv('test.csv')  
sample_submission = pd.read_csv('sample_submission.csv')  

2. Data Cleaning & Null Check  
print(train.isnull().sum())  
train = train.dropna()  

3. Feature Engineering  
train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})  
train['Vehicle_Damage'] = train['Vehicle_Damage'].map({'Yes': 1, 'No': 0})  
train['Vehicle_Age'] = train['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})  

4. Campaign Opportunity Analysis  
import matplotlib.pyplot as plt  
summary_df = train.groupby(['Age', 'Vehicle_Damage']).size().unstack().fillna(0).astype(int)  
summary_df.plot(kind='bar', stacked=True)  
plt.title("Vehicle Damage Distribution by Age")  
plt.show()  

5. SQL-like Querying  
top_channels = train.groupby('Policy_Sales_Channel')['Annual_Premium'].mean().nlargest(5)  
print(top_channels)  

6. Predictive Modeling  
from sklearn.model_selection import train_test_split  
from xgboost import XGBClassifier  
from sklearn.metrics import classification_report  
from sklearn.preprocessing import LabelEncoder  
label_encoder = LabelEncoder()  
for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:  
    train[col] = label_encoder.fit_transform(train[col])  
X = train.drop(['id', 'Response'], axis=1)  
y = train['Response']  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  
model.fit(X_train, y_train)  
y_pred = model.predict(X_val)  
print(classification_report(y_val, y_pred))  

7. Automated Submission Output  
test['Gender'] = test['Gender'].map({'Male': 1, 'Female': 0})  
test['Vehicle_Damage'] = test['Vehicle_Damage'].map({'Yes': 1, 'No': 0})  
test['Vehicle_Age'] = test['Vehicle_Age'].map({'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0})  
X_test = test.drop(['id'], axis=1).fillna(0)  
predictions = model.predict(X_test)  
sample_submission['Response'] = predictions  
sample_submission.to_csv('submission.csv', index=False)  

8. Excel Export for BI  
pivot = train.pivot_table(values='Annual_Premium', index='Age', columns='Vehicle_Damage', aggfunc='mean')  
pivot.to_excel('annual_premium_by_age_and_damage.xlsx')  

Challenges Faced: class imbalance, consistent encoding, categorical data mapping, high accuracy but low recall on positive class.

Dataset: Source – Kaggle (Health Insurance Cross-Sell Prediction). Train samples: 381,109. Test samples: 127,037. Target: Response (1 = interested, 0 = not interested).

Model Outcomes:  
Model – XGBoostClassifier  
Accuracy – ~87%  
Recall – Low for class 1  
Use Case – Insurance campaign automation  

Future Work – AGI/LLM Enhancement: Personalized targeting via LLM-generated content, campaign strategy adaptation through real-time feedback, multi-channel delivery integration via APIs.

References:  
1. Anmol Kumar. Health Insurance Cross-Sell Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction  
2. McKinney, Wes. Python for Data Analysis  
3. Chen, T., & Guestrin, C. XGBoost: A Scalable Tree Boosting System  
4. Pedregosa, F., et al. Scikit-learn: Machine Learning in Python  
5. Hunter, J. D. Matplotlib  
6. Harris, C. R., et al. NumPy  
7. Han, J., & Kamber, M. Data Mining: Concepts and Techniques  
8. Provost, F., & Fawcett, T. Data Science for Business  
9. Kimball, R., & Ross, M. The Data Warehouse Toolkit  
10. Goertzel, B. Artificial General Intelligence  
11. OpenAI. GPT-4 Technical Report. https://openai.com/research/gpt-4  

License: MIT License
