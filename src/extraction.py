import re
from pprint import pprint

def extract_patient_info(text):
    
    patient_info = {}
    patient_name = re.search(r'Patient:\s*([^,]+)', text)
    if patient_name:
        patient_info['name'] = patient_name.group(1).strip()

    patient_age = re.search(r'Age\s+(\d+)', text)
    if patient_age:
        patient_info['age'] = int(patient_age.group(1).strip())

    patient_gender = re.search(r'Gender\s+(\w+)', text)
    if patient_gender:
        patient_info['gender'] = patient_gender.group(1).strip()

    patient_id = re.search(r'ID\s+(\w+)', text)
    if patient_id:  
        patient_info['id'] = patient_id.group(1).strip()

    consulting_hospital = re.search(r'Hospital:\s*(.+?)(?=\n|$)', text)
    if consulting_hospital:
        patient_info['hospital'] = consulting_hospital.group(1).strip()

    consulting_doctor = re.search(r'Consulting Doctor:\s*Dr\.\s*([^,]+)', text)
    if consulting_doctor:
        patient_info['doctor'] = consulting_doctor.group(1).strip()
    
    date = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', text)
    if date:
        patient_info['date'] = date.group(1).strip()

    return patient_info


def find_test_values(text):
    entities = []

    for match in re.finditer(r'\b\d+\.?\d*\b', text):
        start = match.start()
        end = match.end()
        entities.append((start, end, "TEST_VALUE"))
    
    return entities

def find_units(text):
    entities = []
    units = ['g/dL', 'mill/cmm', '/uL', '%', 'mg']
    
    for unit in units:
        for match in re.finditer(re.escape(unit), text):
            start = match.start()
            end = match.end()
            entities.append((start, end, "UNIT"))
    
    return entities


def find_test_names(text):
    """Find positions of test names."""
    entities = []
    test_names = [
        'Haemoglobin', 'Haematocrit', 'Total RBC', 'RBC', 
        'WBC', 'Platelets', 'Neutrophils', 'Lymphocytes', 
        'Monocytes', 'Eosinophils'
    ]
    
    for test in test_names:
        for match in re.finditer(re.escape(test), text, re.IGNORECASE):
            start = match.start()
            end = match.end()
            entities.append((start, end, "TEST_NAME"))
    
    return entities


def find_flags(text):
    """Find positions of abnormality flags."""
    entities = []
    
    pattern = r'Marked as ([HL])\b'
    for match in re.finditer(pattern, text):
        start = match.start(1)
        end = match.end(1)
        entities.append((start, end, "FLAG"))
    
    return entities

def create_training_example(text):
    """Create one training example in spaCy format."""
    entities = []
    
    entities.extend(find_test_names(text))
    entities.extend(find_test_values(text))
    entities.extend(find_units(text))
    entities.extend(find_flags(text))
    
    # Sort by position
    entities = remove_overlaps(entities)

    entities.sort(key=lambda x: x[0])
    
    return (text, {"entities": entities})

def generate_training_data(reports_dict):
    """Generate training data from all reports."""
    training_data = []

    for filename, text in reports_dict.items():
        example = create_training_example(text)
        training_data.append(example)

    return training_data

def remove_overlaps(entities):
    """Remove overlapping entities, keeping longer ones."""
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered = []
    last_end = -1
    
    for start, end, label in entities:
        if start >= last_end:
            filtered.append((start, end, label))
            last_end = end
    
    return filtered

def extract_diagnosis(text):
    """Extract diagnosis from text."""
    diagnosis_match = re.search(r'Diagnosis includes:\s*(.+)', text)
    if diagnosis_match:
        diagnoses = diagnosis_match.group(1).strip()
        return [d.strip() for d in diagnoses.split(',')]
    return []

def extract_medications(text):
    """Extract medications from text."""
    medications = []
    
    in_meds_section = False
    for line in text.split('\n'):
        if 'Medications prescribed:' in line:
            in_meds_section = True
            continue
        
        if in_meds_section:
            if line.strip().startswith('-'):
                med_match = re.search(r'-\s*(.+?)\s+(\d+\s*mg),\s*(\w+)', line)
                if med_match:
                    medications.append({
                        "drug": med_match.group(1).strip(),
                        "dose": med_match.group(2).strip(),
                        "frequency": med_match.group(3).strip()
                    })
            elif line.strip() and not line.strip().startswith('-'):
                break
    
    return medications


def extract_lab_results_ner(text, nlp_model):
    """Extract lab results using trained NER model + regex for flags."""
    lines = text.split('\n')
    lab_results = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        doc = nlp_model(line)
        
        lab = {}
        for ent in doc.ents:
            if ent.label_ == "TEST_NAME" and "test" not in lab:
                lab["test"] = ent.text
            
            elif ent.label_ == "TEST_VALUE" and "value" not in lab:
                try:
                    lab["value"] = float(ent.text)
                except ValueError:
                    pass
            elif ent.label_ == "UNIT" and "unit" not in lab:
                lab["unit"] = ent.text.rstrip('.')
            elif ent.label_ == "FLAG":
                lab["flag"] = ent.text
        
        if lab.get("test"):
            flag_match = re.search(r'Marked as ([HL])', line)
            if flag_match:
                lab["flag"] = flag_match.group(1)
            
            ref_match = re.search(r'compared to normal ([\d.-]+)', line)
            if ref_match:
                lab["reference"] = ref_match.group(1).rstrip('.')
            
            lab_results.append(lab)
    
    return lab_results

def extract_all(text, nlp_model):
    return {
        "patient": extract_patient_info(text),
        "labs": extract_lab_results_ner(text, nlp_model),
        "diagnosis": extract_diagnosis(text),
        "medications": extract_medications(text)
    }

text = """Hospital: Flores, Willis and Doyle Hospital
Patient: Alexis Vance, ID HSP13997, Age 24, Gender Female
Consulting Doctor: Dr. Alexander Miller, Date: 2025-11-15
Some values may vary depending on lab equipment calibration.
Haemoglobin (g/dL) came out to be 8.63 g/dL, compared to normal 12.0-16.0. Marked as L.
Observed Total RBC (mill/cmm): 3.92 mill/cmm. Marked as L.
Haematocrit (%) was measured at 48.59 %. Marked as H.
WBC (/uL) came out to be 11458.19 /uL, compared to normal 4000-10000. Marked as H.
Platelets (/uL) was measured at 142757.37 /uL. Marked as L.
Lab recorded Neutrophils (%) value of 40.85%.
Lab recorded Lymphocytes (%) value of 18.28%. Marked as L.
Monocytes (%) was measured at 4.16 %.
Eosinophils (%) came out to be 0.84 %, compared to normal 1-6. Marked as L.
Some values may vary depending on lab equipment calibration.
Final Clinical Notes:
Diagnosis includes: Iron Deficiency Anemia, Hypertension
Medications prescribed:
 - Metformin 500 mg, BD
 - Atorvastatin 20 mg, HS
Doctor advised proper rest and hydration."""

#info = extract_patient_info(text)
#pprint(info)


sample = "Haemoglobin (g/dL) came out to be 8.63 g/dL, compared to normal 12.0-16.0. Marked as L."
entities = find_test_names(sample)
entities.extend(find_test_values(sample))
entities.extend(find_units(sample))
entities.extend(find_flags(sample))
entities.sort(key=lambda x: x[0])

    
for start, end, label in entities:
    print(f"{label}: '{sample[start:end]}'")
    