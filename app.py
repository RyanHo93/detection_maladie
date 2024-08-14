import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Charger le modèle
model_path = 'C:/Users/ryanh/OneDrive/Documents/ClassifHeart/model_heart.pkl'
model = joblib.load(model_path)

# Titre de l'application
st.title("Prédiction de Maladie Cardiaque")

# Description de l'application
st.write("""
Veuillez entrer les informations ci-dessous pour effectuer une prédiction.
""")

# Saisie des informations utilisateur
age = st.number_input("Âge", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sexe", options={"0": "Femme", "1": "Homme"})
chest_pain_type = st.selectbox("Type de douleur thoracique", options=["TA", "ATA", "NAP", "ASY"])
resting_bp = st.number_input("Pression artérielle au repos (mm Hg)", min_value=0, max_value=300, value=120)
cholesterol = st.number_input("Cholestérol (mg/dL)", min_value=0, max_value=600, value=200)
fasting_bs = st.selectbox("Glycémie à jeun", options={"0": "Inférieure à 120 mg/dL", "1": "Supérieure ou égale à 120 mg/dL"})
resting_ecg = st.selectbox("Résultats de l'ECG au repos", options=["Normal", "ST", "LVH"])
max_hr = st.number_input("Fréquence cardiaque maximale atteinte (bpm)", min_value=0, max_value=220, value=150)
exercise_angina = st.selectbox("Angine induite par l'exercice", options={"Y": "Oui", "N": "Non"})
oldpeak = st.number_input("Dépression ST induite par l'exercice (mm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope = st.selectbox("Pente du segment ST", options=["Up", "Flat", "Down"])

# Encodage des valeurs de l'utilisateur
sex = int(sex)
fasting_bs = int(fasting_bs)
exercise_angina = 1 if exercise_angina == "Y" else 0
chest_pain_type_dict = {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}
chest_pain_type = chest_pain_type_dict[chest_pain_type]
resting_ecg_dict = {"Normal": 0, "ST": 1, "LVH": 2}
resting_ecg = resting_ecg_dict[resting_ecg]
st_slope_dict = {"Up": 0, "Flat": 1, "Down": 2}
st_slope = st_slope_dict[st_slope]

# Création d'un DataFrame pour les features
features = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    
    # Affichage des résultats
    if prediction == 1:
        st.write(f"Le modèle prédit que le patient est susceptible d'avoir une maladie cardiaque avec une probabilité de {prob:.2f}.")
    else:
        st.write(f"Le modèle prédit que le patient n'a probablement pas de maladie cardiaque avec une probabilité de {prob:.2f}.")
