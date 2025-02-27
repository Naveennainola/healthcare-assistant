import torch
import whisper
from django.shortcuts import render, redirect
from django.http import JsonResponse
import re
import ffmpeg
import os
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt
import json
from dotenv import load_dotenv
import joblib
import pandas as pd
from .nlp_processor import extract_symptoms  # ✅ Now used properly
from .disease_predictor import predict_disease  # Import the function
import csv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'feedback.csv')

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["User_Symptoms", "Predicted_Disease", "Correct_Disease"])
# Load Whisper model
try:
    model = whisper.load_model("tiny")
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")

# Load trained model & symptom list
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
mlb_path = os.path.join(MODEL_DIR, "mlb.pkl")

try:
    mlb = joblib.load(mlb_path)
    symptoms_list = sorted(mlb.classes_)  # Get all available symptoms
    print("✅ Symptoms loaded successfully!")
except Exception as e:
    print(f"❌ Error loading symptoms: {e}")
    symptoms_list = []

# Load CSV files for descriptions & precautions
description_path = os.path.join(os.path.dirname(__file__), "dataset", "symptom_Description.csv")
precaution_path = os.path.join(os.path.dirname(__file__), "dataset", "symptom_precaution.csv")

df_description = pd.read_csv(description_path)
df_precaution = pd.read_csv(precaution_path)

# Convert CSVs into dictionaries
disease_descriptions = dict(zip(df_description["Disease"], df_description["Description"]))

disease_precautions = {}
for _, row in df_precaution.iterrows():
    disease = row["Disease"]
    precautions = row.drop("Disease").dropna().tolist()  # Convert to list, remove NaN
    disease_precautions[disease] = precautions

def index(request):
    """
    Render the index page with the symptom dropdown list.
    """
    return render(request, 'assistant/index.html', {'symptoms': symptoms_list})

@csrf_exempt
def speech_to_text(request):
    """
    Convert speech to text using Whisper model and extract medical symptoms using NER.
    """
    if request.method == "POST":
        if "speech_file" not in request.FILES:
            return JsonResponse({"error": "No audio file received"}, status=400)

        speech_file = request.FILES["speech_file"]
        file_path = "temp_audio.wav"
        with open(file_path, "wb") as f:
            for chunk in speech_file.chunks():
                f.write(chunk)

        try:
            result = model.transcribe(file_path)
            transcribed_text = result["text"].strip()

            if not transcribed_text:
                return JsonResponse({"error": "No speech detected"}, status=400)

            # ✅ Use the NER model to extract symptoms from voice input
            extracted_symptoms = extract_symptoms(transcribed_text)
            return JsonResponse({"speech_text": ", ".join(extracted_symptoms)})

        except Exception as e:
            return JsonResponse({"error": "Transcription failed"}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)

def analyze_symptoms(request):
    """
    Analyze user input symptoms, predict disease, and provide first-aid & possible causes.
    """
    if request.method == "POST":
        symptoms = request.POST.get("symptoms", "").split(",")

        if not symptoms or symptoms == [""]:
            return render(request, "assistant/results.html", {"error": "No symptoms detected. Please enter symptoms."})

        predictions = predict_disease(symptoms)

        if "error" in predictions:
            return render(request, "assistant/results.html", {"error": predictions["error"]})

        # Get description & precautions for the most likely disease
        top_disease = predictions[0]["disease"]
        disease_description = disease_descriptions.get(top_disease, "No description available.")
        precautions = disease_precautions.get(top_disease, ["No specific precautions found."])

        # ✅ Pass all diseases to results.html for feedback dropdown
        all_diseases = list(disease_descriptions.keys())
        print("All diseases passed to template:", all_diseases)

        # ✅ Store session variable to track valid results
        request.session['valid_result'] = True

        return render(request, "assistant/results.html", {
            "symptoms": symptoms,
            "predictions": predictions,
            "description": disease_description,
            "precautions": precautions,
            "all_diseases": all_diseases  # ✅ Pass all diseases
        })

    return redirect('index')  # Redirect if accessed incorrectly

def results(request):
    """
    Prevent users from revisiting old results.
    """
    if not request.session.get('valid_result', False):
        return redirect('index')  # Redirect to home if no valid session

    request.session['valid_result'] = False  # Reset after viewing once
    return render(request, "assistant/results.html")

def submit_feedback(request):
    """
    Save user feedback on incorrect predictions.
    """
    if request.method == "POST":
        print("✅ Feedback form submitted!")

        user_symptoms = request.POST.get("user_symptoms", "")
        predicted_disease = request.POST.get("predicted_disease", "")
        correct_disease = request.POST.get("correct_disease", "")

        print(f"User Symptoms: {user_symptoms}")
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Correct Disease: {correct_disease}")

        if correct_disease and correct_disease != predicted_disease:
            # Append feedback to CSV file
            with open(FEEDBACK_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([user_symptoms, predicted_disease, correct_disease])

        return redirect('index')

    print("❌ Invalid request method")
    return redirect('index')
