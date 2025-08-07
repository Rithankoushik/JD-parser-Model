import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import time
import re
import json5
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

@st.cache_resource(show_spinner="Loading model and tokenizer from Hugging Face Hub...")
def load_model_and_tokenizer():
    MODEL_REPO = "Rithankoushik/job-parser-model-qwen"  # your HF repo

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()

def extract_json_from_output(text):
    # Improved JSON extraction: find first '{' and match until the closing '}'
    start = text.find('{')
    if start == -1:
        return text
    stack = []
    for i in range(start, len(text)):
        if text[i] == '{':
            stack.append('{')
        elif text[i] == '}':
            stack.pop()
            if not stack:
                return text[start:i+1]
    # fallback if no matching closing brace found
    return text[start:]

@st.cache_data
def get_static_prompt_parts():
    system_prompt = (
        "You are a highly accurate JSON extractor for job descriptions. "
        "Your ONLY task is to extract what is explicitly mentioned in the job description. "
        "Do NOT guess or infer. If a field is not present in the job description, return an empty value for it. "
        "Always follow the provided JSON schema. Return ONLY the raw JSON object, with no additional text or formatting. "
        "Avoid hallucinations. Do not fabricate emails, phone numbers, websites, salaries, or skills that are not clearly mentioned. "
    )

    json_schema = """{
  "job_titles": [],
  "organization": { "employers": [], "websites": [] },
  "job_contact_details": { "email_address": [], "phone_number": [], "websites": [] },
  "location": { "hiring": [], "org_location": [] },
  "employment_details": { "employment_type": [], "work_mode": [] },
  "compensation": {
    "salary": [
      {
        "amount_in_text": "",
        "time_frequency": "",
        "parsed": { "min": "", "max": "", "currency": "" }
      }
    ],
    "benefits": []
  },
  "technical_skills": [ { "skill_name": "" } ],
  "soft_skills": [],
  "work_experience": {
    "min_in_years": null,
    "max_in_years": null,
    "role_experience": [
      { "min_in_years":null, "max_in_years":null, "skill": "" }
    ],
    "skill_experience": [
      { "min_in_years":null, "max_in_years":null, "skill": "" }
    ]
  },
  "qualifications": [
    { "qualification": [], "specilization": [] }
  ],
  "certifications": [],
  "languages": []
}"""
    example_jd = """Job Title: Sustainability Analyst
Company: HelioCore Energy GmbH
Location:

Hiring for: Berlin, Germany

Org HQ: Berlin, Germany

Employment Type: Full-time
Work Mode: Hybrid (3 days onsite, 2 remote)

Overview:
HelioCore Energy GmbH is at the forefront of Europe's green transition, delivering scalable renewable energy projects across solar, wind, and hydrogen. 
As a Sustainability Analyst, you will work with our ESG, operations, and strategy teams to measure, improve, and report our sustainability performance while staying compliant with EU regulations.

Key Responsibilities:

Collect and analyze sustainability KPIs and ESG metrics from internal teams and partners.

Create dashboards and reports aligned with CSRD and EU Taxonomy compliance.

Collaborate with engineering teams to assess environmental impact of ongoing projects.

Contribute to corporate sustainability strategy and annual disclosures.

Benchmark company initiatives against global sustainability standards (GRI, SASB).

Qualifications & Requirements:

Bachelor's degree in Environmental Science, Sustainability, Economics, or related field.

Up to 2 years of experience in sustainability reporting or ESG analytics.

Proficiency in Excel, Power BI, or similar data tools is a plus.

Familiarity with EU climate policy and frameworks.

Certifications:

GRI Certified Sustainability Professional (preferred)

Languages:

English (Fluent)
German 
Compensation & Benefits:
Salary: €3,000 - €3,600 per month
Benefits: Green mobility stipend, learning budget, hybrid work flexibility, subsidized lunches, gym membership.
Contact Information:
Email: careers@heliocore.de""" 

    example_json_output = """{
  "job_titles": ["Sustainability Analyst"],
  "organization": {
    "employers": ["HelioCore Energy GmbH"],
    "websites": []
  },
  "job_contact_details": {
    "email_address": ["careers@heliocore.de"],
    "phone_number": [],
    "websites": []
  },
  "location": {
    "hiring": ["Berlin, Germany"],
    "org_location": ["Berlin, Germany"]
  },
  "employment_details": {
    "employment_type": ["Full-time"],
    "work_mode": ["Hybrid"]
  },
  "compensation": {
    "salary": [
      {
        "amount_in_text": "€3,000 - €3,600 per month",
        "time_frequency": "monthly",
        "parsed": {
          "min": "3000",
          "max": "3600",
          "currency": "EUR"
        }
      }
    ],
    "benefits": [
      "Green mobility stipend",
      "Learning budget",
      "Hybrid work flexibility",
      "Subsidized lunches",
      "Gym membership"
    ]
  },
  "technical_skills": [
    {"skill_name": "Sustainability reporting"},
    {"skill_name": "ESG metrics"},
    {"skill_name": "Data visualization"},
    {"skill_name": "EU Taxonomy"},
    {"skill_name": "Environmental impact analysis"},
    {"skill_name": "Power BI"},
    {"skill_name": "Excel"},
    {"skill_name": "Carbon footprint modeling"}
  ],
  "soft_skills": [
    "Analytical thinking",
    "Communication",
    "Attention to detail",
    "Team collaboration",
    "Problem-solving"
  ],
  "work_experience": {
    "min_in_years": 0,
    "max_in_years": 2,
    "role_experience": [
      {
        "min_in_years": 0,
        "max_in_years": 2,
        "skill": "Sustainability analytics"
      }
    ],
    "skill_experience": [
      {
        "min_in_years": 0,
        "max_in_years": 2,
        "skill": "ESG frameworks"
      },
      {
        "min_in_years": 0,
        "max_in_years": 1,
        "skill": "Dashboarding"
      }
    ]
  },
  "qualifications": [
    {
      "qualification": ["Bachelor's Degree"],
      "specilization": ["Environmental Science", "Sustainability", "Economics"]
    }
  ],
  "certifications": ["GRI Certified Sustainability Professional"],
  "languages": ["English", "German"]
}""" 


    return system_prompt, json_schema, example_jd, example_json_output


def infer_from_text(jd_text: str):
    start_time = time.time()

    system_prompt, json_schema, example_jd, example_json_output = get_static_prompt_parts()

    # Build user prompt only (changing part)
    user_prompt = f"""

Now, perform the same task on the following new job description.

New Job Description to be parsed:
---
{jd_text}
---

JSON Schema to follow:
---
{json_schema}
---
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=800, do_sample=False)
    raw_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    cleaned = extract_json_from_output(raw_response).replace("None", "null").strip()

    try:
        parsed = json5.loads(cleaned)
    except Exception:
        try:
            parsed = json5.loads(cleaned)
        except Exception:
            return raw_response, round(time.time() - start_time, 2)

    return json.dumps(parsed, indent=2), round(time.time() - start_time, 2)
