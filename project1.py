import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('energydata_complete.csv', index_col=0)

# Data preprocessing
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature Engineering
df_filled.index = pd.to_datetime(df_filled.index)
df_filled['hour_of_day'] = df_filled.index.hour
df_filled['day_of_week'] = df_filled.index.dayofweek
df_filled['month'] = df_filled.index.month
rolling_mean_window = 3
df_filled['T1_rolling_mean'] = df_filled['T1'].rolling(window=rolling_mean_window).mean()
df_filled['T2_multiply_RH2'] = df_filled['T2'] * df_filled['RH_2']
df_filled['is_winter'] = (df_filled.index.month.isin([12, 1, 2])).astype(int)
df_filled['is_spring'] = (df_filled.index.month.isin([3, 4, 5])).astype(int)
df_filled['is_summer'] = (df_filled.index.month.isin([6, 7, 8])).astype(int)
df_filled['is_fall'] = (df_filled.index.month.isin([9, 10, 11])).astype(int)
df_filled = df_filled.dropna()

# Binary Target Variable
threshold = 60 
df_filled['target_binary'] = (df_filled['T_out'] > threshold).astype(int)

# Features (X) and target variable (y)
X = df_filled[['Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']]
y = df_filled['target_binary']

# Data preprocessing
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_imputed_scaled = scaler.fit_transform(imputer.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X_imputed_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Bagging Classifier
bagging_model = BaggingClassifier(base_estimator=best_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)

# AdaBoost Classifier
adaboost_model = AdaBoostClassifier(base_estimator=best_model, n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)

# Model Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n", conf_matrix)

# Save the trained model to a file
joblib.dump(best_model, 'best_model.pkl')

# Load the saved model
loaded_model = joblib.load('best_model.pkl')

def predict(data):
    # Implement logic to preprocess data and make predictions using the loaded model
    # Return the model predictions
    predictions = loaded_model.predict(data)
    return predictions
