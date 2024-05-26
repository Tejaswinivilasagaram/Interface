from flask import Flask, render_template, request
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load trained models
old_rf_model = pickle.load(open('models/model_old.pkl', 'rb'))
middle_rf_model = pickle.load(open('models/model_middle.pkl', 'rb'))
teenage_rf_model = pickle.load(open('models/model_teenage.pkl', 'rb'))
children_rf_model = pickle.load(open('models/model_children.pkl', 'rb'))

# Define features and target columns for each dataset
old_features = ['old_medications', 'old_water', 'old_sleep_patterns', 'old_height', 'old_weight', 'old_numberof_meals', 'old_family_history', 'old_alcohol', 'old_smoke', 'old_loss_of_muscle_mass', 'old_mobility_issues', 'old_age', 'old_gender', 'old_stress', 'old_poor_diet', 'old_heart_disease', 'old_asthma', 'old_diabetes', 'old_economic_constraints', 'old_isolation', 'old_digestive_issues', 'old_hypertension', 'old_chronic_pain']
middle_features = ['middle_water', 'middle_physical_activty', 'middle_height', 'middle_weight', 'middle_technology', 'middle_numberof_meals', 'middle_family_history', 'middle_alcohol', 'middle_transportation', 'middle_sedentary_job', 'middle_smoke', 'middle_screen_time', 'middle_age', 'middle_gender', 'middle_gym_regularly', 'middle_stress', 'middle_heart_disease', 'middle_asthma', 'middle_poor_diet', 'middle_diabetes', 'middle_hypertension', 'middle_marrital_status']
teenage_features = ['teen_water', 'teen_height', 'teen_weight', 'teen_numberof_meals', 'teen_family_history', 'teen_age', 'teen_gender', 'teen_sleep_patterns', 'teen_diabetes', 'teen_financial_constraints', 'teen_vaccination', 'teen_physically_challenged', 'teen_medications', 'teen_early_intake_of_solid_foods', 'teen_environmental_issues', 'teen_clinical_visit', 'teen_lack_ofnutritional_education', 'teen_poor_diet', 'teen_hypertension', 'teen_asthma', 'teen_digestive_issues', 'teen_gym_regularly']
children_features = ['water', 'height', 'weight', 'numberof_meals', 'family_history', 'age', 'gender', 'sleep_patterns', 'diabetes', 'financial_constraints', 'vaccination', 'physically_challenged', 'medications', 'early_intake_of_solid_foods', 'environmental_issues', 'clinical_visit', 'lack_ofnutritional_education', 'digestive_issues', 'stress', 'asthma', 'poor_diet', 'mobility_issues']

# Define target column
target_column = 'class_level'

# Define age categories
age_categories = ['Old', 'Middle', 'Teenage', 'Children']

# Define features for each age category
age_features = {
    'Old': old_features,
    'Middle': middle_features,
    'Teenage': teenage_features,
    'Children': children_features
}

@app.route('/')
def index():
    return render_template('index.html', age_categories=age_categories)

@app.route('/features', methods=['POST'])
def features():
    selected_age = request.form['age_category']
    print("Selected Age Category:", selected_age)  # Add this line
    features = age_features[selected_age]
    return render_template('features.html', selected_age=selected_age, features=features)

@app.route('/predict', methods=['POST'])
def predict():
    age_category = request.form['age_category']
    features = age_features[age_category]
    feature_values = [float(request.form[feature]) for feature in features]

    if age_category == 'Old':
        model = old_rf_model
    elif age_category == 'Middle':
        model = middle_rf_model
    elif age_category == 'Teenage':
        model = teenage_rf_model
    else:
        model = children_rf_model

    prediction = model.predict([feature_values])[0]
    result_message = ""

    if prediction == 0:
        result_message = "You are obese. Please take necessary precautions."
    elif prediction == 1:
        result_message = "Congratulations! You are not obese."

    return render_template('predict.html', age_category=age_category, result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)
