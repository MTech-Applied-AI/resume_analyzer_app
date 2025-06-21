from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple
import uvicorn
import tempfile
import os
import fitz
import docx2txt
from sentence_transformers import SentenceTransformer, util
import requests
import re
import logging
import json
import asyncio
from dotenv import load_dotenv
import boto3
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import botocore.exceptions
import hashlib

from sentence_transformers import CrossEncoder



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

cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant setup
qdrant = QdrantClient(host="localhost", port=6333)
QDRANT_COLLECTION = "similar_matching_resumes"
qdrant.recreate_collection(QDRANT_COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))


# MinIO setup
s3 = boto3.client(
    's3',
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name="us-east-1"
)
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "resumes")
try:
    s3.create_bucket(Bucket=MINIO_BUCKET)
    logger.info(f"Bucket '{MINIO_BUCKET}' created successfully.")
except botocore.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'BucketAlreadyOwnedByYou':
        logger.info(f"Bucket '{MINIO_BUCKET}' already exists and is owned by you.")
    else:
        logger.error(f"Failed to create bucket '{MINIO_BUCKET}': {str(e)}")
        raise e

def compute_file_hash(file: UploadFile) -> str:
    file.file.seek(0)
    sha = hashlib.sha256()
    while chunk := file.file.read(8192):
        sha.update(chunk)
    file.file.seek(0)
    return sha.hexdigest()

def is_duplicate_resume(resume_hash: str) -> bool:
    search = qdrant_client.scroll(
        collection_name="resumes",
        scroll_filter={"must": [{"key": "resume_hash", "match": {"value": resume_hash}}]},
        limit=1
    )
    return bool(search[0])


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


async def save_resume_to_minio(resume: UploadFile, resume_id: str):
    logger.info("Saving resume file to MinIO.")
    resume.file.seek(0)
    s3.upload_fileobj(resume.file, MINIO_BUCKET, f"{resume_id}/{resume.filename}")

async def save_vector_to_qdrant(resume_id: str, vector: List[float], metadata: dict):
    logger.info("Saving vector to Qdrant.")
    point = PointStruct(id=resume_id, vector=vector, payload=metadata)
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=[point])


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


def encode_full_document(text, model, chunk_size=256, stride=128):
    words = text.split()
    if not words:
        return torch.zeros(model.get_sentence_embedding_dimension())

    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), stride)]
    chunk_embeddings = model.encode(
        chunks, convert_to_tensor=True, normalize_embeddings=True
    )
    return chunk_embeddings.mean(dim=0)


@app.post("/upload")
async def analyze_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        logger.info(f"Received resume upload: {resume.filename}")
        filename = resume.filename.lower()
        resume_id = str(uuid.uuid4())
        resume_hash = compute_file_hash(resume)

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

        resume_vec2 = encode_full_document(resume_text, model)
        jd_vec2 = encode_full_document(job_description, model)
        score_chunked = (resume_vec2 @ jd_vec2.T).item() * 100

        logger.info(f"Original Score: {score:.2f}, Chunked Score: {score_chunked:.2f}")
        final_score = max(score, score_chunked)


        logger.info("Calling LLM for resume and JD analysis.")
        resume_skills, jd_skills, exp_summary, education, improvements = await ollama_extract_all(resume_text, job_description)

        # Store to MinIO and Qdrant
        await save_resume_to_minio(resume, resume_id)
        await save_vector_to_qdrant(resume_id, resume_vec.tolist(), {
            "filename": resume.filename,
            "skills": resume_skills,
            "resume_hash": resume_hash,
            "education": education,
            "experience_summary": exp_summary,
            "score": round(final_score, 2)
        })

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


@app.post("/match")
async def match_resumes(job_description: str = Form(...), top_k: int = 10):
    try:
        logger.info("Received match request for job description.")

        jd_vector = model.encode(job_description, convert_to_tensor=True).tolist()

        logger.info("Querying Qdrant for most similar resumes.")
        search_results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=jd_vector,
            limit=top_k
        )

        logger.info(f"Found {len(search_results)} matches before deduplication.")
        matched_resumes = []
        seen_hashes = set()

        for hit in search_results:
            metadata = hit.payload or {}
            resume_hash = metadata.get("resume_hash")
            if resume_hash in seen_hashes:
                continue
            seen_hashes.add(resume_hash)

            resume_id = str(hit.id)
            filename = metadata.get("filename", "unknown")

            try:
                url = s3.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': MINIO_BUCKET,
                        'Key': f"{resume_id}/{filename}"
                    },
                    ExpiresIn=3600
                )
            except Exception as e:
                logger.warning(f"Could not generate presigned URL for {filename}: {e}")
                url = None

            matched_resumes.append({
                "resume_id": resume_id,
                "filename": filename,
                "score": metadata.get("score", 0.0),
                "skills": metadata.get("skills", []),
                "education": metadata.get("education", ""),
                "experience_summary": metadata.get("experience_summary", ""),
                "download_url": url
            })

        logger.info(f"Returning {len(matched_resumes)} unique resumes after filtering duplicates.")
        return {"matches": matched_resumes}

    except Exception as e:
        logger.exception(f"Error during resume matching: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during resume matching.")
