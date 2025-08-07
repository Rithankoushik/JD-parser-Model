import streamlit as st
import json
import torch
from inference import infer_from_text

# GPU check
if torch.cuda.is_available():
    st.info(f" GPU is available: {torch.cuda.get_device_name(0)}")
else:
    st.warning(" GPU is NOT available. Running on CPU.")

# Page config
st.set_page_config(
    page_title="Job Description Parser Demo",
    page_icon="📝",
    layout="wide"
)

# Title
st.markdown("## 📝 Job Description Parser Demo")

# Sample job descriptions
sample_jds = {
    " Machine Learning Engineer Example": """Job Title: Machine Learning Engineer 
About the Role:
At ZentrixAI, we're redefining how data-driven intelligence powers products in healthcare and insurance.
 We're looking for a Machine Learning Engineer to build, train, and optimize models that turn messy real-world data into actionable insights. 
 If you love solving complex problems, deploying scalable ML pipelines, and shipping features that matter, you'll thrive here.

Responsibilities:

Design and develop machine learning models for NLP, tabular prediction, and anomaly detection.

Preprocess and normalize large-scale structured and unstructured datasets.

Collaborate with MLOps to deploy models into production (TensorFlow Serving / TorchServe).

Evaluate model performance using AUC, precision-recall, F1, etc.

Work closely with Data Engineers and Product Managers to define model goals.

Continuously improve models using online learning and feedback loops.

Write scalable training and inference code using TensorFlow and PyTorch.

Maintain model versioning using MLflow and integrate with CI/CD pipelines.

Technical Skills:

Python (NumPy, Pandas, Scikit-learn)

TensorFlow, PyTorch, Keras

MLflow, Docker, FastAPI

SQL, Spark

Cloud ML tools (GCP AI Platform, AWS SageMaker)

NLP libraries (spaCy, Transformers, NLTK)

Git, GitHub Actions, Kubernetes basics

Soft Skills:

Team collaboration

Curiosity and continuous learning

Communication with non-tech stakeholders

Time prioritization

Initiative-taking mindset

Qualifications:

Bachelor's degree in Computer Science, AI, Data Science, or similar

Preferred: Master's in Machine Learning or Applied Mathematics

Certifications:

TensorFlow Developer Certificate

AWS Certified Machine Learning - Specialty

Languages:

English (Fluent)

Mandarin (Basic)

Compensation & Benefits:

Salary: SGD 7,500 - SGD 10,000 per month

Time Frequency: Monthly

Benefits: Remote work setup budget, flexible hours, learning allowance, stock grants, health insurance

Employment Details:

Full-time

Remote (preferably working in Singapore Standard Time)

Location:

Hiring: Remote (Singapore time zone overlap)

Org Location: Singapore

Contact Info:

Email: jobs@zentrixai.com

Phone: +65 6904 8899

Website: https://www.zentrixai.com/careers

About ZentrixAI:
ZentrixAI is an award-winning AI-first company focused on transforming decision-making for insurers and hospitals through intelligent automation. 
With a growing international team, we blend academic rigor with product agility.
"""
}

# Input section
selected = st.selectbox(
    "Select a sample JD to auto-fill the text area", 
    [""] + list(sample_jds.keys())
)
jd_text = st.text_area(
    "Job Description:", 
    value=sample_jds.get(selected, ""), 
    height=300
)

# Parse button and output
if st.button("⚡ Click here to Parse") and jd_text.strip():
    try:
        with st.spinner("Parsing job description..."):
            parsed_output, duration = infer_from_text(jd_text)
        st.success(f"✅ Parsed in {duration} seconds")
        # Try to parse and display as JSON
        try:
            parsed_json = json.loads(parsed_output)
            st.json(parsed_json)
            st.download_button(
                "📋 Download JSON",
                json.dumps(parsed_json, indent=2),
                file_name="parsed_jd.json",
                mime="application/json"
            )
        except Exception:
            st.error("Could not parse output as JSON. Showing raw output:")
            st.code(parsed_output, language="text")
    except Exception as e:
        st.error(f"Error during parsing: {str(e)}")