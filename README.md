---
## 📦 Qwen3-0.6B — Job Description Struct-Extractor

A fine-tuned version of **Qwen3-0.6B** designed for **accurate extraction of structured job attributes** from raw job descriptions. Outputs perfectly schema-aligned JSON — ideal for downstream use in search, analytics, and recommendation systems.

---

### 🚀 Model Highlights

* **Base Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
* **Architecture**: Decoder-only Transformer (Causal Language Model)
* **Tokenizer**: `QwenTokenizer` (same as base)
* **Fine-Tuned For**: Zero-hallucination, schema-conformant information extraction

---

### 🎯 Task Overview

**Task**: Extract structured information from job descriptions
**Output Format**: Strict JSON following a predefined schema
**Use Cases**:

* Automated JD parsing into structured fields
* Building search/match systems for talent platforms
* HR data cleaning & analytics pipelines
* Resume/job matching engines

---

### 🧪 Example Usage (via `transformers`)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Rithankoushik/job-parser-model-qwen-2.0"  # or your HF repo

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

def get_structured_jd(jd_text):
    system_prompt = (
        "You are an expert JSON extractor specifically trained to parse job descriptions into a structured JSON format using a given schema. "
        "Your ONLY goal is to extract exactly and only what is explicitly stated in the job description text. "
        "Do NOT guess, infer, or add any information that is not mentioned. "
        "If a field is not present in the job description, fill it with empty or null values as specified by the schema. "
        "Always perfectly follow the provided JSON schema. "
        "Return ONLY the JSON object with no extra commentary or formatting."
    )

    schema = '''{
      "job_titles": [],
      "organization": { "employers": [], "websites": [] },
      "job_contact_details": { "email_address": [], "phone_number": [], "websites": [] },
      "location": { "hiring": [], "org_location": [] },
      "employment_details": { "employment_type": [], "work_mode": [] },
      "compensation": {
        "salary": [{
          "amount_in_text": "",
          "time_frequency": "",
          "parsed": { "min": "", "max": "", "currency": "" }
        }],
        "benefits": []
      },
      "technical_skills": [{ "skill_name": "" }],
      "soft_skills": [],
      "work_experience": {
        "min_in_years": null,
        "max_in_years": null,
        "role_experience": [{ "min_in_years": null, "max_in_years": null, "skill": "" }],
        "skill_experience": [{ "min_in_years": null, "max_in_years": null, "skill": "" }]
      },
      "qualifications": [{ "qualification": [], "specilization": [] }],
      "certifications": [],
      "languages": []
    }'''

    prompt = f"""
Please extract all explicitly stated information from the following job description and format it as per the JSON schema provided.

Job Description:
\"\"\"
{jd_text}
\"\"\"

JSON Schema:
{schema}

Return ONLY the JSON object.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1200, do_sample=False)

    response = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response

# Example
jd = """
Job Title: Machine Learning Engineer  
Company: ZentrixAI  
Location: Remote (Singapore timezone preferred)  
Salary: SGD 7,500 - 10,000 monthly  
"""

print(get_structured_jd(jd))
```

---

### 🧠 Training Details

* **Data**: Mix of real and synthetic job descriptions from multiple industries and regions (IN/EU/US/Remote)
* **Objective**: Strict extraction without hallucination
* **Labels**: JSON schema covering key job-related fields (titles, skills, compensation, location, etc.)
* **Prompting Strategy**: Instruction-tuned with schema enforcement

---
