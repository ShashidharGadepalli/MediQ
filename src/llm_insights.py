from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()

def generate_diagnostic_insights(patient_data, api_key):
    """Generate diagnostic insights from structured patient data."""
    
    client = OpenAI(api_key=api_key)
    
    abnormal_labs = [lab for lab in patient_data.get('labs', []) if lab.get('flag') in ['H', 'L']]
    
    if not abnormal_labs:
        return {"analysis": "All laboratory values are within normal range."}
    
    prompt = f"""Patient has these ABNORMAL lab values:
{json.dumps(abnormal_labs, indent=2)}

Diagnosis: {', '.join(patient_data.get('diagnosis', []))}

Provide BRIEF clinical insights (3-5 bullet points max):
- What each abnormality might indicate
- Overall pattern interpretation
- How it relates to the diagnosis

Be concise. Each bullet point should be 1 sentence only.
Format as simple bullet points, not paragraphs."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    
    return {"analysis": response.choices[0].message.content}
def add_insights_to_extracted_data():
    """Add LLM insights to existing extracted data."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        return
    
    input_file = "output/extracted_patient_info.json"   # Read from original
    output_file = "output/extracted_patients_with_ai.json"  # Save to new file
    
    with open(input_file, 'r') as f:
        all_patients = json.load(f)
    
    print(f"Generating insights for {len(all_patients)} patients...\n")
    
    for patient_id, patient_data in all_patients.items():
        print(f"Analyzing {patient_id}...")
        insights = generate_diagnostic_insights(patient_data, api_key)
        patient_data['ai_insights'] = insights
    
    with open(output_file, 'w') as f:
        json.dump(all_patients, f, indent=4)
    
    print(f"\nInsights added and saved to {output_file}")

if __name__ == "__main__":
    add_insights_to_extracted_data()