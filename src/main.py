# import json
# import os
# import spacy
# from preprocessing import read_reports_from_folder
# from extraction import extract_all

# def main():
#     folder_path = "data/Train"
#     output_folder = "output"
    
#     os.makedirs(output_folder, exist_ok=True)
    
#     print("Loading NER model...")
#     nlp = spacy.load("medical_ner_model")
    
#     print(f"Loading reports from {folder_path}...")
#     reports = read_reports_from_folder(folder_path, limit=5)
#     print(f"✓ Loaded {len(reports)} reports\n")
    
#     all_patients = {}
    
#     for filename, text in reports.items():
#         print(f"Processing: {filename}")
        
#         complete_data = extract_all(text, nlp)
#         patient_id = complete_data['patient'].get('id', filename)
#         all_patients[patient_id] = complete_data
    
#     output_file = os.path.join(output_folder, "extracted_patient_info.json")
#     with open(output_file, "w") as f:
#         json.dump(all_patients, f, indent=4)
    
#     print(f"\nSaved complete data to: {output_file}")
#     print(f"Total patients processed: {len(all_patients)}")

# if __name__ == "__main__":
#     main()

import json

with open("output/extracted_patients_with_ai.json", "r") as f:
    data = json.load(f)

# Get first patient
first_patient = list(data.values())[0]

# Show AI insights structure
print("AI Insights keys:")
print(first_patient.get('ai_insights', {}).keys())