from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple
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
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logger = logging.getLogger("resume-analyzer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('all-MiniLM-L6-v2')


async def extract_text_from_pdf(upload_file: UploadFile) -> str:
    logger.info("Extracting text from PDF.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)
        text = " ".join([page.get_text() for page in doc])
        return text
    finally:
        doc.close()
        os.unlink(tmp_path)


async def extract_text_from_docx(upload_file: UploadFile) -> str:
    logger.info("Extracting text from DOCX.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path)
    finally:
        os.unlink(tmp_path)


async def ollama_extract_all(resume_text: str, jd_text: str, max_retries: int = 3) -> Tuple[List[str], List[str], str, str, List[str]]:
    logger.info("Calling Groq API for structured extraction.")
    prompt = f"""
        You are an intelligent resume analyzer.

        Return only a valid JSON object with the following fields:
        - resume_skills: list of technical/professional skills found in the resume
        - jd_skills: list of skills required as per the job description
        - experience_summary: a 2-3 line summary of experience
        - education: 1-2 line summary of educational background
        - improvements: list of skills the candidate should add to match the JD

        Resume:
        \"\"\"
        {resume_text}
        \"\"\"

        Job Description:
        \"\"\"
        {jd_text}
        \"\"\"
        Respond ONLY with a valid JSON object. Do NOT include any explanation or markdown formatting.
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Groq API attempt {attempt}")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"Groq response failed with status {response.status_code}")
                continue

            raw = response.json()["choices"][0]["message"]["content"].strip()
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(cleaned)

            logger.info("Successfully parsed Groq response.")
            return (
                data.get("resume_skills", []),
                data.get("jd_skills", []),
                data.get("experience_summary", ""),
                data.get("education", ""),
                data.get("improvements", [])
            )

        except (json.JSONDecodeError, requests.RequestException) as e:
            logger.error(f"Attempt {attempt} failed: {str(e)}")
            if attempt == max_retries:
                logger.error("Max retries reached. Returning empty structured output.")
                return [], [], "", "", []
            await asyncio.sleep(1)

    return [], [], "", "", []


@app.post("/upload")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        logger.info(f"Received resume upload: {resume.filename}")
        filename = resume.filename.lower()

        if filename.endswith(".pdf"):
            resume_text = await extract_text_from_pdf(resume)
        elif filename.endswith(".docx"):
            resume_text = await extract_text_from_docx(resume)
        else:
            logger.error("Unsupported file format")
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and DOCX are allowed.")

        logger.info("Calculating semantic similarity.")
        resume_vec = model.encode(resume_text, convert_to_tensor=True)
        jd_vec = model.encode(job_description, convert_to_tensor=True)
        score = util.cos_sim(resume_vec, jd_vec).item() * 100

        logger.info("Calling LLM for resume and JD analysis.")
        resume_skills, jd_skills, exp_summary, education, improvements = await ollama_extract_all(resume_text, job_description)

        logger.info("Successfully analyzed resume.")
        return {
            "score": round(score, 2),
            "skills": resume_skills,
            "experience_summary": exp_summary,
            "education": education,
            "improvements": improvements
        }

    except Exception as e:
        logger.exception(f"Error during resume analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during resume analysis.")


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
