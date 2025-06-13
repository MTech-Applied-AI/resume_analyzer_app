from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import tempfile
import os
import fitz  # PyMuPDF
import docx2txt
from sentence_transformers import SentenceTransformer, util
import requests
import re
import logging
import json
from typing import List, Tuple
from dotenv import load_dotenv


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load SBERT model for semantic matching
model = SentenceTransformer('all-MiniLM-L6-v2')


async def extract_text_from_pdf(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        text = " ".join([page.get_text() for page in doc])
        doc.close()
        return text
    finally:
        os.unlink(tmp_path)


async def extract_text_from_docx(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path)
    finally:
        os.unlink(tmp_path)


async def ollama_extract_all(resume_text: str, jd_text: str, max_retries: int = 3) -> Tuple[List[str], List[str], str, str, List[str]]:
    prompt = f"""
        Analyze the following resume and job description to extract key information:

        Resume:
        {resume_text}

        Job Description:
        {jd_text}

        Provide a structured analysis in JSON format with the following fields:
        - resume_skills: List of technical and professional skills from the resume
        - jd_skills: List of required skills from the job description
        - experience_summary: Brief summary of professional experience
        - education: Educational background
        - improvements: List of specific suggestions for missing skills

        Format the response as valid JSON only.
    """

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30
            )

            if response.status_code == 200:
                raw = response.json()["choices"][0]["message"]["content"].strip()
                cleaned = raw.strip().strip("```json").strip("```").strip()
                data = json.loads(cleaned)

                return (
                    data.get("resume_skills", []),
                    data.get("jd_skills", []),
                    data.get("experience_summary", ""),
                    data.get("education", ""),
                    data.get("improvements", [])
                )

        except (json.JSONDecodeError, requests.RequestException) as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to extract after {max_retries} retries: {e}")
                return [], [], "", "", []
            await asyncio.sleep(1)

    return [], [], "", "", []


@app.post("/upload")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    logging.info("Resume upload received.")
    filename = resume.filename.lower()

    if filename.endswith(".pdf"):
        resume_text = await extract_text_from_pdf(resume)
    elif filename.endswith(".docx"):
        resume_text = await extract_text_from_docx(resume)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and DOCX are allowed.")

    # Semantic Similarity
    resume_vec = model.encode(resume_text, convert_to_tensor=True)
    jd_vec = model.encode(job_description, convert_to_tensor=True)
    score = util.cos_sim(resume_vec, jd_vec).item() * 100

    # Unified model extraction
    resume_skills, jd_skills, exp_summary, education, improvements = await ollama_extract_all(resume_text, job_description)

    return {
        "score": round(score, 2),
        "skills": resume_skills,
        "experience_summary": exp_summary,
        "education": education,
        "improvements": improvements
    }



# Extract text from PDF
# async def extract_text_from_pdf(upload_file: UploadFile) -> str:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(await upload_file.read())
#         tmp_path = tmp.name

#     try:
#         doc = fitz.open(tmp_path)
#         text = " ".join([page.get_text() for page in doc])
#         doc.close()
#         return text
#     finally:
#         os.unlink(tmp_path)


# # Extract text from DOCX
# async def extract_text_from_docx(upload_file: UploadFile) -> str:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
#         tmp.write(await upload_file.read())
#         tmp_path = tmp.name
#     try:
#         text = docx2txt.process(tmp_path)
#         return text
#     finally:
#         os.unlink(tmp_path)

# # Extract skills from text using Ollama + Mistral
# def extract_skills_ollama(text: str) -> List[str]:
#     prompt = f"""
#     Extract all professional skills, tools, programming languages, technologies, and certifications from the following resume text. Return them as a JSON list of strings only. Do not add explanation.

#     """
#     {text}
#     """
#     """

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "mistral",
#             "prompt": prompt,
#             "stream": False
#         }
#     )
#     if response.status_code == 200:
#         try:
#             output = response.json()["response"]
#             skills = eval(output.strip())  # Ensure output is a valid list
#             return skills if isinstance(skills, list) else []
#         except Exception as e:
#             print("Ollama parsing error:", e)
#             return []
#     else:
#         print("Ollama API error:", response.text)
#         return []


# def ollama_extract_field(text: str, field: str) -> str:
#     prompt = f"""
#     Extract only the {field} information from the resume text below. 
#     Respond with a single sentence or short paragraph, without any explanation or formatting.

#     Resume Text:
#     \"\"\"
#     {text}
#     \"\"\"
#     """
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "mistral", "prompt": prompt, "stream": False}
#     )
#     if response.status_code == 200:
#         try:
#             output = response.json()["response"].strip()
#             return output
#         except Exception as e:
#             print(f"Ollama parsing error for {field}:", e)
#             return f"{field} not found"
#     else:
#         print(f"Ollama API error while extracting {field}:", response.text)
#         return f"{field} not found"

# def generate_suggestions_from_missing_skills(missing_skills: List[str]) -> List[str]:
#     prompt = f"""
#     Given the following missing skills in a candidate's resume:
#     {missing_skills}

#     Suggest professional, constructive recommendations (one per skill) to help the candidate improve their profile.
#     Return the suggestions as a numbered list (string output).
#     """

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "mistral", "prompt": prompt, "stream": False}
#     )
#     if response.status_code == 200:
#         try:
#             return response.json()["response"].strip().split("\n")
#         except Exception as e:
#             print("Ollama suggestion parsing error:", e)
#             return []
#     else:
#         print("Ollama API error in suggestions:", response.text)
#         return []
    

# @app.post("/upload")
# async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
#     logging.info("Received request for resume analysis")
#     filename = resume.filename.lower()
#     logging.info(f"Uploaded file: {filename}")

#     if filename.endswith(".pdf"):
#         logging.info("Processing PDF resume...")
#         resume_text = await extract_text_from_pdf(resume)
#     elif filename.endswith(".docx"):
#         logging.info("Processing DOCX resume...")
#         resume_text = await extract_text_from_docx(resume)
#     else:
#         logging.warning("Unsupported file format received.")
#         raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and DOCX are allowed.")

#     # Semantic Similarity
#     logging.info("Generating vector embeddings for similarity scoring...")
#     resume_vec = model.encode(resume_text, convert_to_tensor=True)
#     jd_vec = model.encode(job_description, convert_to_tensor=True)
#     score = util.cos_sim(resume_vec, jd_vec).item() * 100
#     logging.info(f"Match Score Computed: {score:.2f}%")

#     # Skill Extraction
#     logging.info("Extracting skills using Ollama from resume...")
#     skills = extract_skills_ollama(resume_text)
#     logging.info(f"Skills extracted from resume: {skills}")

#     logging.info("Extracting skills using Ollama from job description...")
#     jd_skills = extract_skills_ollama(job_description)
#     logging.info(f"Skills extracted from JD: {jd_skills}")

#     # Education & Experience
#     # education = extract_education(resume_text)
#     education = ollama_extract_field(resume_text, "education")
#     logging.info(f"Education extracted: {education}")

#     # exp_summary = extract_experience_summary(resume_text)
#     exp_summary = ollama_extract_field(resume_text, "professional experience summary")
#     logging.info(f"Experience summary extracted: {exp_summary}")

#     # Skill Gap Analysis
#     improvements = []
#     missing_skills = list(set(jd_skills) - set(skills))
#     # for skill in missing_skills:
#     #     improvements.append(f"Consider adding experience with '{skill}' to better align with the job description.")
#     # logging.info(f"Suggested improvements: {improvements}")
#     improvements = generate_suggestions_from_missing_skills(missing_skills)

#     logging.info("Returning final analysis result to frontend.")
#     return {
#         "score": round(score, 2),
#         "skills": skills,
#         "experience_summary": exp_summary,
#         "education": education,
#         "improvements": improvements
#     }