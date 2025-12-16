import os
import uuid
from pathlib import Path
from typing import List, Optional

import lance
import pyarrow as pa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI(title="Lance Vector DB - PDF Search")

# Configuration
LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", "./lance_data")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Initialize embedding model
embedding_model = SentenceTransformer(MODEL_NAME)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

# Lance dataset reference
dataset: Optional[lance.Dataset] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str
    document_name: str
    page_number: int
    score: float


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks


def extract_text_from_pdf(file_path: str) -> List[dict]:
    """Extract text from PDF and return chunks with metadata."""
    reader = PdfReader(file_path)
    chunks_with_metadata = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            chunks = chunk_text(text)
            for chunk in chunks:
                if chunk.strip():
                    chunks_with_metadata.append({
                        "text": chunk.strip(),
                        "page_number": page_num,
                    })
    
    return chunks_with_metadata


def initialize_dataset():
    """Initialize or load the Lance dataset."""
    global dataset
    
    Path(LANCE_DB_PATH).mkdir(parents=True, exist_ok=True)
    dataset_path = Path(LANCE_DB_PATH) / "documents"
    
    if dataset_path.exists():
        try:
            dataset = lance.dataset(str(dataset_path))
            print(f"Loaded existing dataset with {len(dataset)} records")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            dataset = None
    else:
        dataset = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    initialize_dataset()
    print(f"Embedding model: {MODEL_NAME} (dim: {EMBEDDING_DIM})")


@app.get("/")
async def root():
    """Health check endpoint."""
    record_count = len(dataset) if dataset else 0
    return {
        "status": "healthy",
        "service": "Lance Vector DB - PDF Search",
        "records": record_count,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    global dataset
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text chunks from PDF
        chunks_with_metadata = extract_text_from_pdf(temp_path)
        
        if not chunks_with_metadata:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks_with_metadata]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        
        # Prepare data for Lance
        data = {
            "id": [str(uuid.uuid4()) for _ in chunks_with_metadata],
            "text": texts,
            "document_name": [file.filename] * len(chunks_with_metadata),
            "page_number": [chunk["page_number"] for chunk in chunks_with_metadata],
            "vector": embeddings.tolist(),
        }
        
        # Create PyArrow table
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("document_name", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ])
        table = pa.Table.from_pydict(data, schema=schema)
        
        # Write to Lance dataset
        dataset_path = Path(LANCE_DB_PATH) / "documents"
        if dataset is None:
            # Create new dataset
            lance.write_dataset(table, str(dataset_path), mode="create")
            dataset = lance.dataset(str(dataset_path))
        else:
            # Append to existing dataset
            lance.write_dataset(table, str(dataset_path), mode="append")
            dataset = lance.dataset(str(dataset_path))
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded and indexed {file.filename}",
                "chunks_added": len(chunks_with_metadata),
                "total_records": len(dataset)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Search for semantically similar text chunks."""
    global dataset
    
    if dataset is None:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet")
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode([request.query], show_progress_bar=False)[0]
        
        # Perform vector search
        results = (
            dataset.scanner(nearest={"column": "vector", "q": query_embedding.tolist(), "k": request.top_k})
            .to_table()
            .to_pydict()
        )
        
        # Format results
        search_results = []
        for i in range(len(results["text"])):
            search_results.append(SearchResult(
                text=results["text"][i],
                document_name=results["document_name"][i],
                page_number=results["page_number"][i],
                score=float(results.get("_distance", [0])[i]) if "_distance" in results else 0.0
            ))
        
        return search_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")


@app.delete("/reset")
async def reset_database():
    """Delete all indexed documents."""
    global dataset
    
    try:
        dataset_path = Path(LANCE_DB_PATH) / "documents"
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)
            dataset = None
            return {"message": "Database reset successfully"}
        else:
            return {"message": "No database to reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
