import spacy
import os

# Define the correct path to the extracted model
MODEL_PATH = r"C://Users//Sai Kumar//spacy_models//en_ner_bc5cdr_md"  # Windows
# MODEL_PATH = "/home/yourname/spacy_models/en_core_sci_sm"  # Linux/Mac

# Ensure the path exists before loading
if os.path.exists(MODEL_PATH):
    try:
        nlp = spacy.load(MODEL_PATH)  # Load model from corrected path
        print("✅ SciSpaCy model loaded successfully!")
    except Exception as e:
        print(f"❌ SciSpaCy loading failed: {e}")
        nlp = None
else:
    print(f"❌ Model path not found: {MODEL_PATH}")
    nlp = None

def extract_symptoms(text):
    """
    Extracts symptoms from user input using SciSpaCy.
    """
    if nlp is None:
        print("⚠️ Error: SciSpaCy model is not loaded.")
        return []

    try:
        doc = nlp(text.lower())
        symptoms = set()

        # Identify symptoms & diseases from medical entities
        for ent in doc.ents:
            if ent.label_ in ["DISEASE"]:
                symptoms.add(ent.text)

        return list(symptoms)

    except Exception as e:
        print(f"⚠️ SciSpaCy extraction error: {e}")
        return []
