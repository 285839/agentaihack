# app.py

import streamlit as st
import json

# Load mock data
with open("frontend/mock_data.json", "r") as f:

    past_cases = json.load(f)

st.set_page_config(page_title="NeuroSync.Health", layout="wide")
st.title("🧠 NeuroSync.Health – Agentic Clinical Decision Mirror")

st.markdown("This prototype simulates how an AI agent team supports clinical decisions by matching patient cases, predicting outcomes, and checking ethical concerns.")

# Patient input form
# Patient input form in centered column
st.header("📋 New Patient Case Input")

# Create 3 columns and use the middle one
col1, col2, col3 = st.columns([1, 2, 1])  # Makes center column 2x wider

with col2:
    with st.form("case_input_form"):
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Symptoms (comma-separated)")
        submitted = st.form_submit_button("Analyze Case")


if submitted:
    input_symptoms = [s.strip().lower() for s in symptoms.split(",")]

    # Basic symptom match (simulate agent behavior)
    matched_case = None
    for case in past_cases:
        overlap = set(input_symptoms).intersection(set([s.lower() for s in case["symptoms"]]))
        if len(overlap) >= 1:
            matched_case = case
            break

    st.divider()
    st.subheader("🤖 Agentic Decision Flow")

    # Step 1: Memory Match
    st.markdown("### 🔍 1. Matched Past Case")
    if matched_case:
        st.json(matched_case)
    else:
        st.error("No matching case found. (Add more mock data!)")

    # Step 2: Outcome Simulation
    st.markdown("### 📈 2. Predicted Outcome")
    if matched_case:
        st.success(f"Likely outcome: {matched_case['outcome']}")

    # Step 3: Ethical Check
    st.markdown("### ⚖️ 3. Ethical Alignment Check")
    if matched_case and matched_case["ethical_flags"]:
        st.warning(f"Ethical concern flagged: {', '.join(matched_case['ethical_flags'])}")
    else:
        st.success("No ethical concerns detected.")

    # Step 4: Synthesized Recommendation
    st.markdown("### 🧾 4. Summary Recommendation")
    if matched_case:
        st.info(f"Recommended Treatment: {matched_case['treatment']}\n\nDiagnosis: {matched_case['diagnosis']}")

    st.markdown("### 🧬 Agent Flow")
    st.markdown("""
    > 🧠 Medical Memory → 🔍 Diagnostic Mirror → 📈 Outcome Simulator → ⚖️ Ethical Alignment → ✅ Synthesized Insight
    """)








# import streamlit as st
# import json

# st.set_page_config(page_title="NeuroSync.Health Prototype", layout="wide")

# # --- Load past cases ---
# with open("mock_data.json", "r") as f:
#     past_cases = json.load(f)

# # --- Header ---
# st.title("🧠 NeuroSync.Health – Clinical Decision Mirror")

# # --- Input Form ---
# st.header("📋 New Patient Case Input")
# with st.form("case_input_form"):
#     age = st.number_input("Age", min_value=0)
#     gender = st.selectbox("Gender", ["male", "female", "other"])
#     symptoms = st.text_area("Symptoms (comma-separated)")
#     submitted = st.form_submit_button("Analyze")

# # --- Processing ---
# if submitted:
#     st.subheader("🤖 Agentic Decision Flow")

#     # Simulate matching past case
#     match = past_cases[0]  # Simplified for now
#     st.markdown("**🔍 Matching Past Case:**")
#     st.json(match)

#     # Simulated output
#     st.markdown("**📈 Predicted Outcome:**")
#     st.success("Patient likely to stabilize within 3 days with treatment X.")

#     st.markdown("**⚖️ Ethical Check:**")
#     st.warning("No ethical conflicts detected in recommended treatment.")

#     st.markdown("**🧾 Summary Recommendation:**")
#     st.info(f"Recommend starting treatment: {match['treatment']}. Monitor for 48 hours.")

# # Optional: visual flow (timeline or stepper)
# st.markdown("""
# ### 🧬 Decision Flow:
# 1. 🔍 **Case Input**
# 2. 🧠 **Memory Match**
# 3. 📈 **Outcome Prediction**
# 4. ⚖️ **Ethical Review**
# 5. ✅ **Recommendation**
# """)
