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

# Extract text from PDF
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

# Extract skills from text using Ollama + Mistral
def extract_skills_ollama(text: str) -> List[str]:
    prompt = f"""
    Extract all professional skills, tools, programming languages, technologies, and certifications from the following resume text. Return them as a JSON list of strings only. Do not add explanation.

    """
    {text}
    """
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    if response.status_code == 200:
        try:
            output = response.json()["response"]
            skills = eval(output.strip())  # Ensure output is a valid list
            return skills if isinstance(skills, list) else []
        except Exception as e:
            print("Ollama parsing error:", e)
            return []
    else:
        print("Ollama API error:", response.text)
        return []

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

def ollama_extract_field(text: str, field: str) -> str:
    prompt = f"""
    Extract only the {field} information from the resume text below. 
    Respond with a single sentence or short paragraph, without any explanation or formatting.

    Resume Text:
    \"\"\"
    {text}
    \"\"\"
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        try:
            output = response.json()["response"].strip()
            return output
        except Exception as e:
            print(f"Ollama parsing error for {field}:", e)
            return f"{field} not found"
    else:
        print(f"Ollama API error while extracting {field}:", response.text)
        return f"{field} not found"

def generate_suggestions_from_missing_skills(missing_skills: List[str]) -> List[str]:
    prompt = f"""
    Given the following missing skills in a candidate's resume:
    {missing_skills}

    Suggest professional, constructive recommendations (one per skill) to help the candidate improve their profile.
    Return the suggestions as a numbered list (string output).
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        try:
            return response.json()["response"].strip().split("\n")
        except Exception as e:
            print("Ollama suggestion parsing error:", e)
            return []
    else:
        print("Ollama API error in suggestions:", response.text)
        return []
    

@app.post("/upload")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    logging.info("Received request for resume analysis")
    filename = resume.filename.lower()
    logging.info(f"Uploaded file: {filename}")

    if filename.endswith(".pdf"):
        logging.info("Processing PDF resume...")
        resume_text = await extract_text_from_pdf(resume)
    elif filename.endswith(".docx"):
        logging.info("Processing DOCX resume...")
        resume_text = await extract_text_from_docx(resume)
    else:
        logging.warning("Unsupported file format received.")
        raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and DOCX are allowed.")

    # Semantic Similarity
    logging.info("Generating vector embeddings for similarity scoring...")
    resume_vec = model.encode(resume_text, convert_to_tensor=True)
    jd_vec = model.encode(job_description, convert_to_tensor=True)
    score = util.cos_sim(resume_vec, jd_vec).item() * 100
    logging.info(f"Match Score Computed: {score:.2f}%")

    # Skill Extraction
    logging.info("Extracting skills using Ollama from resume...")
    skills = extract_skills_ollama(resume_text)
    logging.info(f"Skills extracted from resume: {skills}")

    logging.info("Extracting skills using Ollama from job description...")
    jd_skills = extract_skills_ollama(job_description)
    logging.info(f"Skills extracted from JD: {jd_skills}")

    # Education & Experience
    # education = extract_education(resume_text)
    education = ollama_extract_field(resume_text, "education")
    logging.info(f"Education extracted: {education}")

    # exp_summary = extract_experience_summary(resume_text)
    exp_summary = ollama_extract_field(resume_text, "professional experience summary")
    logging.info(f"Experience summary extracted: {exp_summary}")

    # Skill Gap Analysis
    improvements = []
    missing_skills = list(set(jd_skills) - set(skills))
    # for skill in missing_skills:
    #     improvements.append(f"Consider adding experience with '{skill}' to better align with the job description.")
    # logging.info(f"Suggested improvements: {improvements}")
    improvements = generate_suggestions_from_missing_skills(missing_skills)

    logging.info("Returning final analysis result to frontend.")
    return {
        "score": round(score, 2),
        "skills": skills,
        "experience_summary": exp_summary,
        "education": education,
        "improvements": improvements
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Optimized Backend: FastAPI + Ollama (Batch Prompt to Reduce Latency)

# from fastapi import FastAPI, File, Form, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Tuple, Dict, Optional
# import uvicorn
# import tempfile
# import os
# import fitz  # PyMuPDF
# import docx2txt
# from sentence_transformers import SentenceTransformer, util
# import requests
# import json
# import logging
# from functools import lru_cache
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
# import hashlib

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# app = FastAPI()
# executor = ThreadPoolExecutor(max_workers=4)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load SBERT model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Cache for document text extraction
# @lru_cache(maxsize=100)
# def get_cached_text(file_hash: str) -> Optional[str]:
#     return None

# def compute_file_hash(content: bytes) -> str:
#     return hashlib.md5(content).hexdigest()

# # Optimized file parsers with caching
# async def extract_text_from_pdf(upload_file: UploadFile) -> str:
#     content = await upload_file.read()
#     file_hash = compute_file_hash(content)
    
#     # Check cache first
#     cached_text = get_cached_text(file_hash)
#     if cached_text:
#         return cached_text

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(content)
#         tmp_path = tmp.name
#     try:
#         doc = fitz.open(tmp_path)
#         text = " ".join([page.get_text() for page in doc])
#         doc.close()
#         return text
#     finally:
#         os.unlink(tmp_path)

# async def extract_text_from_docx(upload_file: UploadFile) -> str:
#     content = await upload_file.read()
#     file_hash = compute_file_hash(content)
    
#     # Check cache first
#     cached_text = get_cached_text(file_hash)
#     if cached_text:
#         return cached_text

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
#         tmp.write(content)
#         tmp_path = tmp.name
#     try:
#         text = docx2txt.process(tmp_path)
#         return text
#     finally:
#         os.unlink(tmp_path)

# # Optimized LLM call with retries and better error handling
# async def ollama_extract_all(resume_text: str, jd_text: str, max_retries: int = 3) -> Tuple[List[str], List[str], str, str, List[str]]:
#     prompt = f"""
#     Analyze the following resume and job description to extract key information:

#     Resume:
#     {resume_text}

#     Job Description:
#     {jd_text}

#     Provide a structured analysis in JSON format with the following fields:
#     - resume_skills: List of technical and professional skills from the resume
#     - jd_skills: List of required skills from the job description
#     - experience_summary: Brief summary of professional experience
#     - education: Educational background
#     - improvements: List of specific suggestions for missing skills

#     Format the response as valid JSON only.
#     """

#     for attempt in range(max_retries):
#         try:
#             response = requests.post(
#                 "http://localhost:11434/api/generate",
#                 json={
#                     "model": "mistral",
#                     "prompt": prompt,
#                     "stream": False,
#                     "temperature": 0.3  # Lower temperature for more consistent output
#                 },
#                 timeout=30  # Add timeout
#             )
            
#             if response.status_code == 200:
#                 raw = response.json()["response"].strip()
#                 cleaned = raw.strip().strip("```json").strip("```").strip()
#                 data = json.loads(cleaned)
                
#                 return (
#                     data.get("resume_skills", []),
#                     data.get("jd_skills", []),
#                     data.get("experience_summary", ""),
#                     data.get("education", ""),
#                     data.get("improvements", [])
#                 )
#         except (json.JSONDecodeError, requests.RequestException) as e:
#             if attempt == max_retries - 1:
#                 logging.error(f"Failed to process after {max_retries} attempts: {str(e)}")
#                 return [], [], "", "", []
#             await asyncio.sleep(1)  # Wait before retry
    
#     return [], [], "", "", []

# # Optimized endpoint with parallel processing
# @app.post("/upload")
# async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
#     try:
#         logging.info("Received request for resume analysis")
#         filename = resume.filename.lower()
        
#         # Extract text based on file type
#         if filename.endswith(".pdf"):
#             resume_text = await extract_text_from_pdf(resume)
#         elif filename.endswith(".docx"):
#             resume_text = await extract_text_from_docx(resume)
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and DOCX are allowed.")

#         # Process embeddings and LLM analysis in parallel
#         async def process_embeddings():
#             resume_vec = model.encode(resume_text, convert_to_tensor=True)
#             jd_vec = model.encode(job_description, convert_to_tensor=True)
#             return util.cos_sim(resume_vec, jd_vec).item() * 100

#         async def process_llm():
#             return await ollama_extract_all(resume_text, job_description)

#         # Run both tasks concurrently
#         score_task = asyncio.create_task(process_embeddings())
#         llm_task = asyncio.create_task(process_llm())
        
#         score = await score_task
#         skills, jd_skills, exp_summary, education, improvements = await llm_task

#         return {
#             "score": round(score, 2),
#             "skills": skills,
#             "experience_summary": exp_summary,
#             "education": education,
#             "improvements": improvements
#         }

#     except Exception as e:
#         logging.error(f"Error processing resume: {str(e)}")
#         raise HTTPException(status_code=500, detail="Error processing resume")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
