# app.py

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Callable, Dict
import os
import uvicorn
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import io
import json
import httpx

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request model
class AnswerResponse(BaseModel):
    answer: str

# Sample environment variable usage
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
NLP_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Function map to dynamically call the correct function based on question similarity
function_map: Dict[str, Callable] = {}
question_embeddings = {}

# Function registration decorator using question text
def register_question(question_text: str):
    def decorator(func: Callable):
        function_map[question_text] = func
        question_embeddings[question_text] = None  # Placeholder for external embedding
        return func
    return decorator

# ga5 q1 - Calculate total margin from Excel file
@register_question("What is the total margin for transactions before Fri Nov 25 2022 for Theta sold in IN?")
async def calculate_total_margin(file: UploadFile) -> str:
    file_content = await file.read()
    df = pd.read_excel(io.BytesIO(file_content))
    df['Customer Name'] = df['Customer Name'].str.strip()
    country_mapping = {
        "Ind": "IN", "India": "IN",
        "USA": "US", "U.S.A": "US", "US": "US",
        "UK": "GB", "U.K": "GB", "United Kingdom": "GB",
        "Fra": "FR", "France": "FR",
        "Bra": "BR", "Brazil": "BR"
    }
    df['Country'] = df['Country'].str.strip().map(country_mapping).fillna(df['Country'])
    def parse_date(date_str):
        for fmt in ("%Y/%m/%d", "%m-%d-%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    df['Date'] = df['Date'].apply(parse_date)
    df['Product'] = df['Product/Code'].str.split('/').str[0].str.strip()
    df['Sales'] = df['Sales'].str.replace("USD", "").str.replace(" ", "").astype(float)
    df['Cost'] = df['Cost'].str.replace("USD", "").str.replace(" ", "")
    df['Cost'] = df['Cost'].apply(lambda x: float(x) if pd.notnull(x) else None)
    df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)
    filter_date = datetime(2022, 11, 25, 6, 28, 5)
    filtered_df = df[(df['Date'] <= filter_date) & 
                     (df['Product'] == 'Theta') & 
                     (df['Country'] == 'IN')]
    total_sales = filtered_df['Sales'].sum()
    total_cost = filtered_df['Cost'].sum()
    total_margin = (total_sales - total_cost) / total_sales if total_sales != 0 else 0
    return f"{total_margin:.4f}"

# ga1 q1 - Output of 'code -s'
@register_question("What is the output of code -s?")
async def get_code_s_output() -> str:
    return "Cannot execute 'code -s' in a serverless environment. Please run this command locally."

# ga5 q6 - Calculate total sales from JSONL file
@register_question("What is the total sales value?")
async def calculate_total_sales(file: UploadFile) -> str:
    file_content = await file.read()
    total_sales = 0.0
    for line in file_content.splitlines():
        try:
            data = json.loads(line)
            sales_value = float(data.get('sales', 0))
            total_sales += sales_value
        except json.JSONDecodeError:
            continue
    return f"{total_sales:.2f}"

async def get_embedding_from_external_api(text: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": text}]
            }
            response = await client.post(NLP_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json().get("choices")[0]["message"]["content"]
    except Exception as e:
        print(f"Error fetching embedding: {e}")
    return None

@app.post("/api/", response_model=AnswerResponse)
async def get_answer(question: str = Form(...), file: Optional[UploadFile] = None):
    try:
        # Get embedding for the input question
        answer = await get_embedding_from_external_api(question)
        if not answer:
            return AnswerResponse(answer="Failed to get response from AI Proxy.")

        return AnswerResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
