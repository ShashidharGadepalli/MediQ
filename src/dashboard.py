import streamlit as st
import json
import os
st.set_page_config(page_title="MediQ Dashboard", layout="wide")

st.title("🏥 MediQ - Medical Report Dashboard")
st.markdown("---")

data_file = "output/extracted_patients_with_ai.json"

if not os.path.exists(data_file):
    st.error("No data found! Run main.py first to extract data.")
    st.stop()

with open(data_file, 'r') as f:
    all_patients = json.load(f)

patient_ids = list(all_patients.keys())
selected_id = st.sidebar.selectbox("Select Patient ID", patient_ids)

patient_data = all_patients[selected_id]

st.header(f"Patient: {patient_data['patient']['name']}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Age", patient_data['patient']['age'])
with col2:
    st.metric("Gender", patient_data['patient']['gender'])
with col3:
    st.metric("Patient ID", patient_data['patient']['id'])
with col4:
    st.metric("Date", patient_data['patient']['date'])

st.subheader("🏥 Hospital Information")
st.write(f"**Hospital:** {patient_data['patient']['hospital']}")
st.write(f"**Doctor:** Dr. {patient_data['patient']['doctor']}")

st.markdown("---")

st.subheader("🧪 Laboratory Results")

lab_data = patient_data['labs']

for lab in lab_data:
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    
    with col1:
        st.write(f"**{lab['test']}**")
    
    with col2:
        if 'value' in lab:
            st.write(f"{lab['value']} {lab.get('unit', '')}")
        else:
            st.write(f"— {lab.get('unit', '')}")
    
    with col3:
        if 'reference' in lab:
            st.write(f"Normal: {lab['reference']}")
        else:
            st.write("—")
    
    with col4:
        flag = lab.get('flag', '')
        if flag == 'H':
            st.markdown("🔴 **HIGH**")
        elif flag == 'L':
            st.markdown("🔵 **LOW**")
        else:
            st.markdown("✅ Normal")

st.markdown("---")

st.subheader("📋 Diagnosis")
for dx in patient_data['diagnosis']:
    st.write(f"• {dx}")

st.markdown("---")

st.subheader("💊 Medications")
for med in patient_data['medications']:
    st.write(f"• **{med['drug']}** - {med['dose']} ({med['frequency']})")




st.markdown("---")

st.subheader("AI Diagnostic Insights")

if 'ai_insights' in patient_data and patient_data['ai_insights']:
    insights = patient_data['ai_insights']
    
    if isinstance(insights, dict) and 'analysis' in insights:
        st.write(insights['analysis'])
    
    st.caption("Note: AI-generated insights. Not medical advice.")
else:
    st.info("No AI insights available.")