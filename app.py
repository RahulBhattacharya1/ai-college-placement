import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Placement Probability + Salary Band", page_icon="🎓", layout="centered")

@st.cache_resource
def load_pipeline():
    path = Path("models/placement_pipeline.pkl")
    if not path.exists():
        st.error("Model file not found at models/placement_pipeline.pkl")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_band_rules():
    path = Path("band_rules.json")
    if not path.exists():
        st.error("band_rules.json not found.")
        st.stop()
    with open(path, "r") as f:
        return json.load(f)

pipe = load_pipeline()
rules = load_band_rules()

st.title("Placement Probability + Salary Band Recommender")
st.caption("Trained on CollegePlacement.csv (no salary column). Bands are rule-based and configurable.")

with st.expander("How bands are computed", expanded=False):
    st.write("""
    1. We compute **placement probability** using a trained classifier.  
    2. We read thresholds from **band_rules.json**.  
    3. We pick the first band whose minimum criteria are satisfied (probability, CGPA, IQ, Projects).
    """)

# ====== Input form ======
with st.form("input_form", clear_on_submit=False):
    st.subheader("Enter Profile")

    # Inputs aligned with your dataset schema
    # College_ID (string)
    college_id = st.text_input("College ID", value="C001")

    # IQ (int)
    iq = st.number_input("IQ", min_value=50, max_value=200, value=110, step=1)

    # Prev_Sem_Result (float, e.g., percentage)
    prev_sem = st.number_input("Previous Semester Result (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)

    # CGPA (float)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.2, step=0.1)

    # Academic_Performance (int; assume some institutional score 0-100 or 0-10)
    academic_perf = st.number_input("Academic Performance (score)", min_value=0, max_value=100, value=75, step=1)

    # Internship_Experience (Yes/No)
    internship = st.selectbox("Internship Experience", options=["Yes", "No"], index=0)

    # Extra_Curricular_Score (int)
    extra_curr = st.number_input("Extra Curricular Score", min_value=0, max_value=100, value=60, step=1)

    # Communication_Skills (int)
    comms = st.number_input("Communication Skills (score)", min_value=0, max_value=100, value=70, step=1)

    # Projects_Completed (int)
    projects = st.number_input("Projects Completed", min_value=0, max_value=50, value=2, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare a single-row dataframe for the pipeline
    input_row = pd.DataFrame([{
        "College_ID": college_id,
        "IQ": int(iq),
        "Prev_Sem_Result": float(prev_sem),
        "CGPA": float(cgpa),
        "Academic_Performance": int(academic_perf),
        "Internship_Experience": internship,
        "Extra_Curricular_Score": int(extra_curr),
        "Communication_Skills": int(comms),
        "Projects_Completed": int(projects)
    }])

    try:
        proba = pipe.predict_proba(input_row)[:, 1][0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("### Results")
    st.metric("Placement Probability", f"{proba*100:.1f}%")

    # Salary Band Recommendation (rule-based)
    band_name = "Low"
    for band in rules.get("bands", []):
        ok_prob = proba >= band.get("min_prob", 0.0)
        ok_cgpa = cgpa >= band.get("min_cgpa", 0.0)
        ok_iq = iq >= band.get("min_iq", 0)
        ok_proj = projects >= band.get("min_projects", 0)

        if ok_prob and ok_cgpa and ok_iq and ok_proj:
            band_name = band["name"]
            break

    st.success(f"Recommended Salary Band: **{band_name}**")

    with st.expander("Why this band?"):
        st.write({
            "probability": round(float(proba), 4),
            "CGPA": cgpa,
            "IQ": iq,
            "Projects_Completed": projects,
            "applied_rule": band_name
        })

st.divider()
st.caption("Tip: Edit thresholds in band_rules.json on GitHub and redeploy to change band recommendations without retraining.")
