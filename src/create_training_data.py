import json
from preprocessing import read_reports_from_folder
from extraction import generate_training_data

reports = read_reports_from_folder("data/Train", limit=70)
print(f"Loaded {len(reports)} reports")

training_data = generate_training_data(reports)
print(f"Created {len(training_data)} training examples")

with open("training_data.json", "w") as f:
    json.dump(training_data, f, indent=2)

print("Saved training_data.json")