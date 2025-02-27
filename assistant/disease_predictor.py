import joblib
import os
import google.generativeai as genai
import json
from dotenv import load_dotenv
import time
import numpy as np

# Load the trained model and MultiLabelBinarizer
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
model_path = os.path.join(MODEL_DIR, "disease_model.pkl")
mlb_path = os.path.join(MODEL_DIR, "mlb.pkl")

load_dotenv()

try:
    rf_model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)
    print("✅ Disease prediction model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    rf_model = None
    mlb = None

# Load Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def map_symptoms_using_gemini(user_symptoms):
    """
    Use Gemini API to map user symptoms to the closest symptoms in the trained model.
    """
    if not mlb:
        print("❌ Error: MultiLabelBinarizer not loaded!")
        return user_symptoms

    trained_symptoms = list(mlb.classes_)  # Get all known symptoms

    # Prepare a strict prompt for Gemini to ONLY return valid symptoms
    prompt = f"""
    You are a medical assistant. Match each user symptom to the closest symptom from this list:
    {json.dumps(trained_symptoms, indent=2)}

    - If there's a direct match, return it.  
    - If there's a close match, return the closest symptom from the list.  
    - If a symptom is not found, discard it.  
    - DO NOT add new symptoms that are not in the list.  
    - Return ONLY a JSON list of matched symptoms.

    User symptoms: {json.dumps(user_symptoms, indent=2)}
    """

    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        # time.sleep(2)  # Prevent rate limit errors
        response = gemini_model.generate_content(prompt)

        # Ensure response is valid JSON
        if not response.text or response.text.strip() == "":
            print("⚠️ Gemini API returned an empty response! Using original symptoms.")
            return user_symptoms  # Fallback to original symptoms

        try:
            corrected_symptoms = json.loads(response.text)
            if not isinstance(corrected_symptoms, list):
                print("⚠️ Unexpected response format! Using original symptoms.")
                return user_symptoms
        except json.JSONDecodeError:
            print("⚠️ Gemini response is not valid JSON! Using original symptoms.")
            return user_symptoms

        # Ensure all mapped symptoms exist in training data
        valid_corrected_symptoms = [s for s in corrected_symptoms if s in trained_symptoms]

        print("🔍 Gemini-mapped Symptoms:", valid_corrected_symptoms)
        return valid_corrected_symptoms
    except Exception as e:
        print(f"⚠️ Gemini API Error: {e}")
        return user_symptoms  # Fallback if API fails

def predict_disease(symptoms):
    """
    Predict the disease based on extracted symptoms.
    """
    if rf_model is None or mlb is None:
        return {"Error: Model not loaded"}

    try:
        # Map symptoms using Gemini AI
        symptoms = map_symptoms_using_gemini(symptoms)

        # Convert symptoms into the same binary format used in training
        symptom_vector = mlb.transform([symptoms])

        # Debugging: Print vector to check if it's empty
        print("🩺 Symptom Vector:", symptom_vector)

        # Ensure at least one symptom is present
        if symptom_vector.sum() == 0:
            print("❌ No matching symptoms found in training data!")
            return "Error: Symptoms not recognized"

        # Get probability scores for all diseases
        probabilities = rf_model.predict_proba(symptom_vector)[0]
        disease_classes = rf_model.classes_

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_diseases = [(disease_classes[i], probabilities[i]) for i in top_indices]

        results = [{"disease":disease, "confidence":f"{confidence * 100:.2f}%"} for disease, confidence in top_diseases]

        return results
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return "Error in prediction"
