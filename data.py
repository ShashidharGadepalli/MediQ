import random
import os
from faker import Faker
import datetime

fake = Faker()

lab_ranges = {
    "Haemoglobin (g/dL)": (12.0, 16.0),
    "Total RBC (mill/cmm)": (4.2, 5.4),
    "Haematocrit (%)": (36, 46),
    "WBC (/uL)": (4000, 10000),
    "Platelets (/uL)": (150000, 450000),
    "Neutrophils (%)": (40, 70),
    "Lymphocytes (%)": (20, 40),
    "Monocytes (%)": (2, 10),
    "Eosinophils (%)": (1, 6)
}

diagnosis_list = [
    "Type 2 Diabetes Mellitus",
    "Hypertension",
    "Iron Deficiency Anemia",
    "Leukocytosis (Infection)",
    "Coronary Artery Disease"
]

medications_list = [
    ("Metformin", "500 mg", "BD"),
    ("Atorvastatin", "20 mg", "HS"),
    ("Amoxicillin-Clavulanate", "625 mg", "TDS"),
    ("Aspirin", "81 mg", "OD"),
    ("Lisinopril", "10 mg", "OD")
]

# Random filler/noise sentences
noise_sentences = [
    "This report is generated electronically and may not need signature.",
    "Patient reported feeling slightly weak yesterday evening.",
    "Doctor advised proper rest and hydration.",
    "Some values may vary depending on lab equipment calibration.",
    "Report verified by automated system."
]

def generate_lab_results():
    results = []
    for test, (low, high) in lab_ranges.items():
        value = round(random.uniform(low * 0.7, high * 1.3), 2)  # add variation
        unit = test.split("(")[-1].replace(")", "")
        flag = ""
        if value < low:
            flag = "L"
        elif value > high:
            flag = "H"
        phrasing = random.choice([
            f"{test} was measured at {value} {unit}.",
            f"Observed {test}: {value} {unit}.",
            f"{test} came out to be {value} {unit}, compared to normal {low}-{high}.",
            f"Lab recorded {test} value of {value}{unit}.",
        ])
        if flag:
            phrasing += f" Marked as {flag}."
        results.append(phrasing)
    return results

def generate_report_text():
    # Patient details
    name = fake.name()
    age = random.randint(20, 80)
    gender = random.choice(["Male", "Female"])
    patient_id = fake.bothify(text="HSP#####")
    doctor = fake.name()
    hospital = fake.company() + " Hospital"
    date = datetime.date.today().strftime("%Y-%m-%d")

    diagnosis = random.sample(diagnosis_list, k=random.randint(1, 2))
    meds = random.sample(medications_list, k=random.randint(1, 3))
    labs = generate_lab_results()

    # Build unstructured report
    report = []
    report.append(f"Hospital: {hospital}")
    report.append(f"Patient: {name}, ID {patient_id}, Age {age}, Gender {gender}")
    report.append(f"Consulting Doctor: Dr. {doctor}, Date: {date}")
    report.append(random.choice(noise_sentences))
    report.extend(labs)
    report.append(random.choice(noise_sentences))
    report.append("Final Clinical Notes:")
    report.append("Diagnosis includes: " + ", ".join(diagnosis))
    report.append("Medications prescribed:")
    for d, dose, freq in meds:
        report.append(f" - {d} {dose}, {freq}")
    report.append(random.choice(noise_sentences))

    return "\n".join(report)

def generate_dataset(n=10, save=True):
    folder = "data/Train"
    os.makedirs(folder, exist_ok=True)

    data = []
    for i in range(n):
        report_text = generate_report_text()
        data.append(report_text)

        if save:
            with open(os.path.join(folder, f"report_{i+1}.txt"), "w") as f:
                f.write(report_text)

    return data

if __name__ == "__main__":
    dataset = generate_dataset(100)
    print("Generated text reports saved in 'reports_text/' folder")
