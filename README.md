# Lance Vector DB - PDF Semantic Search

A FastAPI service that indexes PDF documents into a Lance vector database and enables semantic search using sentence transformers.

## Features

- 📄 Upload and index PDF documents
- 🔍 Semantic search across all indexed documents
- 🚀 Fast vector similarity search with Lance
- 🐳 Docker-ready for easy deployment
- ☁️ Google Cloud Run compatible

## Local Development

### Prerequisites

- Python 3.11+
- pip

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

The API will be available at `http://localhost:8080`

## API Endpoints

### Health Check
```bash
GET /
```

### Upload PDF
```bash
curl -X POST "http://localhost:8080/upload" \
  -F "file=@document.pdf"
```

### Search
```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 5}'
```

### Reset Database
```bash
curl -X DELETE "http://localhost:8080/reset"
```

## Docker

### Build
```bash
docker build -t lance-vector-db .
```

### Run
```bash
docker run -p 8080:8080 lance-vector-db
```

## Deploy to Google Cloud Run

### Prerequisites
- Google Cloud SDK installed
- Project created on Google Cloud
- Artifact Registry repository created

### Steps

1. Configure gcloud:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. Build and push to Artifact Registry:
```bash
# Set variables
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=us-central1
export SERVICE_NAME=lance-vector-db

# Build for Cloud Run
gcloud builds submit --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME}
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy ${SERVICE_NAME} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars LANCE_DB_PATH=/app/lance_data,EMBEDDING_MODEL=all-MiniLM-L6-v2
```

4. Get the service URL:
```bash
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
```

## Configuration

Environment variables:

- `LANCE_DB_PATH`: Directory for Lance database (default: `./lance_data`)
- `EMBEDDING_MODEL`: HuggingFace model for embeddings (default: `all-MiniLM-L6-v2`)
- `CHUNK_SIZE`: Text chunk size in characters (default: `500`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `50`)
- `PORT`: Server port (default: `8080`)

## Architecture

1. **PDF Upload**: Documents are uploaded, text is extracted, chunked, and embedded
2. **Vector Storage**: Lance stores embeddings with metadata (filename, page number)
3. **Search**: Query is embedded and nearest neighbors are retrieved from Lance
4. **Results**: Returns ranked results with source document and page information

## Technology Stack

- **Lance**: Columnar vector database optimized for ML workloads
- **FastAPI**: Modern Python web framework
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **PyPDF**: PDF text extraction
- **Docker**: Containerization for deployment

## Notes

- The embedding model (`all-MiniLM-L6-v2`) downloads on first run (~80MB)
- Lance database persists in the `lance_data` directory
- For production, consider using persistent storage (e.g., Cloud Storage FUSE)
