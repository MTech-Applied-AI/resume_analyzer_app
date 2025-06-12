# Backend: Production-Ready FastAPI Resume Analyzer

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import tempfile
import os
import fitz  # PyMuPDF
import docx2txt
from sentence_transformers import SentenceTransformer, util
import re
from keybert import KeyBERT



app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SBERT model (semantic-aware and production lightweight)
model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model='all-MiniLM-L6-v2')


# Skill dictionary (can be moved to DB or config file)
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'aws', 'gcp', 'azure', 'docker', 'kubernetes',
    'terraform', 'linux', 'sql', 'nosql', 'mongodb', 'postgresql',
    'git', 'github', 'jenkins', 'ci/cd', 'flask', 'django', 'react', 'node.js'
]

# Extract text from PDF
async def extract_text_from_pdf(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        text = " ".join([page.get_text() for page in doc])
        return text
    finally:
        os.unlink(tmp_path)

# Extract text from DOCX
async def extract_text_from_docx(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path)
        return text
    finally:
        os.unlink(tmp_path)

# Extract skills by keyword matching
def extract_skills(text: str) -> List[str]:
    text = text.lower()
    return list(set([skill for skill in SKILL_KEYWORDS if skill in text]))

def extract_skills_keybert(text: str, top_n: int = 10) -> List[str]:
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

# Extract experience summary heuristically
def extract_experience_summary(text: str) -> str:
    experience_lines = [line for line in text.splitlines() if re.search(r'\d+\+?\s+years', line, re.I)]
    return experience_lines[0].strip() if experience_lines else "Experience information not found."

# Extract education heuristically
def extract_education(text: str) -> str:
    education_keywords = ['b.tech', 'm.tech', 'bachelor', 'master', 'b.sc', 'm.sc', 'phd']
    lines = text.lower().splitlines()
    for line in lines:
        if any(keyword in line for keyword in education_keywords):
            return line.strip()
    return "Education information not found."

@app.post("/upload")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    filename = resume.filename.lower()
    if filename.endswith(".pdf"):
        resume_text = await extract_text_from_pdf(resume)
    elif filename.endswith(".docx"):
        resume_text = await extract_text_from_docx(resume)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and DOCX are allowed.")

    # Encode resume and JD
    resume_vec = model.encode(resume_text, convert_to_tensor=True)
    jd_vec = model.encode(job_description, convert_to_tensor=True)
    score: int | float = util.cos_sim(resume_vec, jd_vec).item() * 100

    # Extract details
    # skills = extract_skills(resume_text)
    skills = extract_skills_keybert(resume_text)
    education = extract_education(resume_text)
    exp_summary = extract_experience_summary(resume_text)

    improvements = []
    jd_skills = extract_skills_keybert(job_description)
    missing_skills = list(set(jd_skills) - set(skills))
    for skill in missing_skills:
        improvements.append(f"Consider adding experience with '{skill}' to better align with the job description.")

    return {
        "score": round(score, 2),
        "skills": skills,
        "experience_summary": exp_summary,
        "education": education,
        "improvements": improvements
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
