from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import requests
import logging
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# FastAPI app setup
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (modify if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Request model
class SearchRequest(BaseModel):
    docs: List[str]
    query: str

# Response model
class SearchResponse(BaseModel):
    matches: List[str]

# Proxy Configuration
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Read from .env

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using the proxy API"""
    try:
        if not AIPROXY_TOKEN:
            raise ValueError("AIPROXY_TOKEN is missing. Set it in your .env file.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": texts,
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    except Exception as e:
        logging.error(f"Error fetching embeddings: {e}")
        raise

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    if not v1_array.any() or not v2_array.any():
        raise ValueError("One or both vectors are empty")
    return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))

@app.post("/similarity", response_model=SearchResponse)
async def get_similar_docs(request: SearchRequest):
    try:
        logging.debug(f"Received request: {request}")

        # Combine docs and query for batch embedding
        all_texts = request.docs + [request.query]
        embeddings = get_embeddings(all_texts)

        # Split embeddings
        doc_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1]

        # Compute similarities
        similarities = [
            (i, cosine_similarity(doc_emb, query_embedding))
            for i, doc_emb in enumerate(doc_embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 matches
        top_matches = [request.docs[idx] for idx, _ in similarities[:3]]
        return SearchResponse(matches=top_matches)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run( "ga3_q7:app", host="0.0.0.0", port=10000, reload=True)
