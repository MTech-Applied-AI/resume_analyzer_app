Backend - uvicorn resume_analyzer_ollama:app --host 0.0.0.0 --port 8000 --reload

Frontend - npm run start

```bash

docker run -d --name minio `
 -p 9000:9000 `
 -p 9001:9001 `
 -e "MINIO_ROOT_USER=minioadmin" `
 -e "MINIO_ROOT_PASSWORD=minioadmin123" `
 -v D:\semester3\nlp_project\resume_analyzer_app\minio:/data `
 minio/minio server /data --console-address ":9001"

```
Distribution on Work
Archita - Mathching logic - similarity macthing, cross encoder matching and contribution to the front end
Abhishek - major contribution in front end, exploring llm models, worked on initial local model mistaraal basic poc
Narayan - worked on averaging logic for similarity matching, find similar match resume, and contribution to the FE and Groq implementatoin
