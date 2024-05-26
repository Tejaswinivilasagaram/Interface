import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pickle

# Load datasets
old_df = pd.read_csv('datasets/OLD.csv')
middle_df = pd.read_csv('datasets/MIDDLE.csv')
teenage_df = pd.read_csv('datasets/TEEN.csv')
children_df = pd.read_csv('datasets/CHILD.csv')

# Define features and target columns for each dataset
old_features = ['old_medications', 'old_water', 'old_sleep_patterns', 'old_height', 'old_weight', 'old_numberof_meals', 'old_family_history', 'old_alcohol', 'old_smoke', 'old_loss_of_muscle_mass', 'old_mobility_issues', 'old_age', 'old_gender', 'old_stress', 'old_poor_diet', 'old_heart_disease', 'old_asthma', 'old_diabetes', 'old_economic_constraints', 'old_isolation', 'old_digestive_issues', 'old_hypertension', 'old_chronic_pain']
middle_features = ['middle_water', 'middle_physical_activty', 'middle_height', 'middle_weight', 'middle_technology', 'middle_numberof_meals', 'middle_family_history', 'middle_alcohol', 'middle_transportation', 'middle_sedentary_job', 'middle_smoke', 'middle_screen_time', 'middle_age', 'middle_gender', 'middle_gym_regularly', 'middle_stress', 'middle_heart_disease', 'middle_asthma', 'middle_poor_diet', 'middle_diabetes', 'middle_hypertension', 'middle_marrital_status']
teenage_features = ['teen_water', 'teen_height', 'teen_weight', 'teen_numberof_meals', 'teen_family_history', 'teen_age', 'teen_gender', 'teen_sleep_patterns', 'teen_diabetes', 'teen_financial_constraints', 'teen_vaccination', 'teen_physically_challenged', 'teen_medications', 'teen_early_intake_of_solid_foods', 'teen_environmental_issues', 'teen_clinical_visit', 'teen_lack_ofnutritional_education', 'teen_poor_diet', 'teen_hypertension', 'teen_asthma', 'teen_digestive_issues', 'teen_gym_regularly']
children_features = ['water', 'height', 'weight', 'numberof_meals', 'family_history', 'age', 'gender', 'sleep_patterns', 'diabetes', 'financial_constraints', 'vaccination', 'physically_challenged', 'medications', 'early_intake_of_solid_foods', 'environmental_issues', 'clinical_visit', 'lack_ofnutritional_education', 'digestive_issues', 'stress', 'asthma', 'poor_diet', 'mobility_issues']

# Define target column
target_column = 'class_level'
# Drop rows with missing target values
old_df.dropna(subset=[target_column], inplace=True)
middle_df.dropna(subset=[target_column], inplace=True)
teenage_df.dropna(subset=[target_column], inplace=True)
children_df.dropna(subset=[target_column], inplace=True)

# Define features and target columns for each dataset after dropping missing target values
old_X = old_df[old_features]
old_y = old_df[target_column]
middle_X = middle_df[middle_features]
middle_y = middle_df[target_column]
teenage_X = teenage_df[teenage_features]
teenage_y = teenage_df[target_column]
children_X = children_df[children_features]
children_y = children_df[target_column]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
old_X = imputer.fit_transform(old_X)
middle_X = imputer.fit_transform(middle_X)
teenage_X = imputer.fit_transform(teenage_X)
children_X = imputer.fit_transform(children_X)

# Train-test split for each dataset
old_X_train, old_X_test, old_y_train, old_y_test = train_test_split(old_X, old_y, test_size=0.2, random_state=42)
middle_X_train, middle_X_test, middle_y_train, middle_y_test = train_test_split(middle_X, middle_y, test_size=0.2, random_state=42)
teenage_X_train, teenage_X_test, teenage_y_train, teenage_y_test = train_test_split(teenage_X, teenage_y, test_size=0.2, random_state=42)
children_X_train, children_X_test, children_y_train, children_y_test = train_test_split(children_X, children_y, test_size=0.2, random_state=42)

# Train a Random Forest model for each dataset
old_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
old_rf_model.fit(old_X_train, old_y_train)
old_accuracy = accuracy_score(old_y_test, old_rf_model.predict(old_X_test))
print("Old Dataset Accuracy:", old_accuracy)

middle_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
middle_rf_model.fit(middle_X_train, middle_y_train)
middle_accuracy = accuracy_score(middle_y_test, middle_rf_model.predict(middle_X_test))
print("Middle Dataset Accuracy:", middle_accuracy)

teenage_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
teenage_rf_model.fit(teenage_X_train, teenage_y_train)
teenage_accuracy = accuracy_score(teenage_y_test, teenage_rf_model.predict(teenage_X_test))
print("Teenage Dataset Accuracy:", teenage_accuracy)

children_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
children_rf_model.fit(children_X_train, children_y_train)
children_accuracy = accuracy_score(children_y_test, children_rf_model.predict(children_X_test))
print("Children Dataset Accuracy:", children_accuracy)

# Save models
pickle.dump(old_rf_model, open('models/model_old.pkl', 'wb'))
pickle.dump(middle_rf_model, open('models/model_middle.pkl', 'wb'))
pickle.dump(teenage_rf_model, open('models/model_teenage.pkl', 'wb'))
pickle.dump(children_rf_model, open('models/model_children.pkl', 'wb'))

print("Models saved successfully!")
