import streamlit as st

st.set_page_config(
    page_title="Platypus — Health Risk Platform",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦆 Platypus Health Risk Platform")
st.markdown("""
Welcome to the **Platypus** patient risk scoring and habit recommendation platform.

Built with NHANES + SDoH data, validated clinical risk equations (PCE, FINDRISC, PLCO),
XGBoost + SHAP explainability, and an LLM-powered habit recommendation engine.

---
### Select a view from the sidebar to get started:

| View | Who it's for | What it shows |
|------|-------------|---------------|
| 🧑 Patient Portal | Patients & families | Personal risk scores + daily habit plan |
| 🩺 Clinician Dashboard | PCPs & care teams | Panel risk flags + SHAP driver breakdown |
| 📊 Insurer Analytics | Health plans | Population risk + preventive ROI model |
""")

st.sidebar.success("Select a view above ↑")
