import spacy
from extraction import extract_lab_results_ner, extract_diagnosis, extract_medications
import json

nlp = spacy.load("medical_ner_model")

sample = """Haemoglobin (g/dL) came out to be 8.63 g/dL, compared to normal 12.0-16.0. Marked as L.
WBC (/uL) was measured at 11458.19 /uL, compared to normal 4000-10000. Marked as H.
Platelets (/uL) was measured at 142757.37 /uL. Marked as L.
Final Clinical Notes:
Diagnosis includes: Iron Deficiency Anemia, Hypertension
Medications prescribed:
 - Metformin 500 mg, BD
 - Atorvastatin 20 mg, HS"""

print("Lab Results:")
results = extract_lab_results_ner(sample, nlp)
print(json.dumps(results, indent=2))

print("\nDiagnosis:")
print(json.dumps(extract_diagnosis(sample), indent=2))

print("\nMedications:")
print(json.dumps(extract_medications(sample), indent=2))