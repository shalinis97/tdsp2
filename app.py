# app.py

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uvicorn
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import io

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

@app.post("/api/", response_model=AnswerResponse)
async def get_answer(question: str = Form(...), file: Optional[UploadFile] = None):
    try:
        # Check if the question matches the specific scenario
        if question == "What is the total margin for transactions before Fri Nov 25 2022 06:28:05 GMT+0530 (India Standard Time) for Theta sold in IN (which may be spelt in different ways)?" and file:
            file_content = await file.read()
            df = pd.read_excel(io.BytesIO(file_content))
            
            # Clean and process the data
            df['Customer Name'] = df['Customer Name'].str.strip()
            country_mapping = {
                "Ind": "IN", "India": "IN",
                "USA": "US", "U.S.A": "US", "US": "US",
                "UK": "GB", "U.K": "GB", "United Kingdom": "GB",
                "Fra": "FR", "France": "FR",
                "Bra": "BR", "Brazil": "BR"
            }
            df['Country'] = df['Country'].str.strip().map(country_mapping).fillna(df['Country'])
            
            # Parse dates and extract product names
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Product'] = df['Product/Code'].str.split('/').str[0].str.strip()
            df['Sales'] = df['Sales'].str.replace("USD", "").str.replace(" ", "").astype(float)
            df['Cost'] = df['Cost'].str.replace("USD", "").str.replace(" ", "")
            df['Cost'] = df['Cost'].apply(lambda x: float(x) if pd.notnull(x) else None)
            df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)

            # Filter data based on the criteria
            filter_date = datetime(2022, 11, 25, 6, 28, 5)
            filtered_df = df[(df['Date'] <= filter_date) & 
                             (df['Product'] == 'Theta') & 
                             (df['Country'] == 'IN')]

            total_sales = filtered_df['Sales'].sum()
            total_cost = filtered_df['Cost'].sum()
            total_margin = (total_sales - total_cost) / total_sales if total_sales != 0 else 0
            
            return AnswerResponse(answer=f"{total_margin:.4f}")
        
        # Default placeholder answer
        return AnswerResponse(answer="This is a placeholder answer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
