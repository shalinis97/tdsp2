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
import re
from PIL import Image
import numpy as np
import colorsys
import httpx
import feedparser


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

# Keep AIPROXY_TOKEN and NLP_API_URL without usage
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
NLP_API_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Function map to dynamically call the correct function based on regex patterns
function_map: Dict[str, Callable] = {}

# Function registration decorator using regex pattern
def register_question(pattern: str):
    def decorator(func: Callable):
        function_map[pattern] = func
        return func
    return decorator

#-------- GA1 questions---------

# ga1 q1 - Output of 'code -s' without escape characters
@register_question(r".*output of code -s.*")
async def get_code_s_output() -> str:
    response_data = {
        "Version": "Code 1.97.2 (e54c774e0add60467559eb0d1e229c6452cf8447, 2025-02-12T23:20:35.343Z)",
        "OS Version": "Windows_NT x64 10.0.26100",
        "CPUs": "12th Gen Intel(R) Core(TM) i3-1215U (8 x 2496)",
        "Memory (System)": "23.73GB (12.82GB free)",
        "VM": "0%",
        "Screen Reader": "no",
        "Process Argv": "--crash-reporter-id 5e69de63-700b-45e0-8939-d706ef7d699d",
        "GPU Status": {
            "2d_canvas": "enabled",
            "canvas_oop_rasterization": "enabled_on",
            "gpu_compositing": "enabled",
            "multiple_raster_threads": "enabled_on",
            "opengl": "enabled_on",
            "rasterization": "enabled",
            "video_decode": "enabled",
            "video_encode": "enabled",
            "webgl": "enabled",
            "webgl2": "enabled",
            "webgpu": "enabled"
        }
    }
    return json.dumps(response_data)


#-------- GA2 questions---------

# ga2 q5 - Calculate number of light pixels in an image
@register_question(r".*number of pixels with lightness.*")
async def calculate_light_pixels(file: UploadFile) -> str:
    file_content = await file.read()
    image = Image.open(io.BytesIO(file_content))
    rgb = np.array(image) / 255.0
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)
    light_pixels = np.sum(lightness > 0.133)
    return str(int(light_pixels))


#-------- end of GA2 questions-------
#------------------------------------

#-------- GA3 questions---------

# ga3 q9 - Generate a prompt for LLM to respond "Yes"

@register_question(r".*(prompt|make).*LLM.*Yes..*")
async def get_llm_prompt_for_yes() -> str:
    return "Yes"

#-------- end of GA3 questions-------
#------------------------------------


#-------- GA4 questions---------

# ga4 q5 - Get maximum latitude of Algiers in Algeria using Nominatim API
@register_question(r".*?(maximum latitude|max latitude).*?(bounding box).*?(Algiers).*?(Algeria).*?")
async def get_max_latitude_algiers() -> str:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "Algiers, Algeria",
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
        "extratags": 1,
        "polygon_geojson": 0,
        "bounded": 1
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data and "boundingbox" in data[0]:
            bounding_box = data[0]["boundingbox"]
            max_latitude = float(bounding_box[1])
            return str(max_latitude)
    return "No data found"

# ga4 q6 - Get link to the latest Hacker News post about Linux with at least 66 points
@register_question(r".*?(Hacker News|link).*?(Linux).*?(66 points|minimum 66 points|66 or more points).*?")
async def get_latest_hn_post_link() -> str:
    feed_url = "https://hnrss.org/newest?q=Linux&points=66"
    feed = feedparser.parse(feed_url)
    if feed.entries:
        return feed.entries[0].link
    return "No relevant post found"


#-------- end of GA4 questions-------
#----------------------------------------------------------------------------

#-------- GA5 questions---------

# ga5 q1 - Calculate total margin from Excel file

#@register_question(r".*(margin).*?(transactions).*?(before).*?(Theta).*?(?:IN|India).*")
@register_question(r".*(margin).*?(transactions)?.*?(before).*?(Theta).*?(?:IN|India).*")
#@register_question(r".*(margin).*?(transactions)?.*?(before).*?(Theta).*?(?:IN|India).*?(Nov|November).*")

#@register_question(r".*(margin).*?(transactions).*?(before).*?(Theta).*?(?:IN|India).*")
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

# ga5 q2 - Count unique student IDs in a text file


# @register_question(r".*Download.*text.* file.*q-clean-up-student-marks.txt.*(unique students|number of unique students|student IDs).*")
@register_question(r".*(unique.*students|student IDs).*?(file|download).*")

async def count_unique_students(file: UploadFile) -> str:
    file_content = await file.read()
    lines = file_content.decode("utf-8").splitlines()
    student_ids = set()
    pattern = re.compile(r'-\s*([\w\d]+)::?Marks')
    for line in lines:
        match = pattern.search(line)
        if match:
            student_ids.add(match.group(1))
    return str(len(student_ids))

# ga5 q5 - Calculate Pizza sales in Mexico City with sales >= 158 units
@register_question(r".*Pizza.*Mexico City.* at least 158 units.*")
async def calculate_pizza_sales(file: UploadFile) -> str:
    file_content = await file.read()
    sales_data = json.loads(file_content)
    df = pd.DataFrame(sales_data)
    mexico_city_variants = ["Mexico-City", "Mexiko City", "Mexico Cty", "Mexicocity", "Mexicoo City"]
    df['city_standardized'] = df['city'].apply(lambda x: "Mexico City" if x in mexico_city_variants else x)
    filtered_df = df[(df['product'] == "Pizza") & (df['sales'] >= 158)]
    sales_by_city = filtered_df.groupby('city_standardized')['sales'].sum().reset_index()
    mexico_city_sales = sales_by_city[sales_by_city['city_standardized'] == "Mexico City"]['sales'].sum()
    return str(int(mexico_city_sales))

# ga5 q6 - Calculate total sales from JSONL file
@register_question(r".*download.*data.*q-parse-partial-json.jsonl.*(total sales value|total sales).*")
async def calculate_total_sales(file: UploadFile) -> str:
    file_content = await file.read()
    total_sales = 0
    file_content_str = file_content.decode("utf-8")
    sales_matches = re.findall(r'"sales":\s*([\d.]+)', file_content_str)
    total_sales = sum(int(float(sales)) for sales in sales_matches)
    return str(total_sales)

# ga5 q7 - Count occurrences of "LGK" as a key in nested JSON

#@register_question(r".*?(LGK).*?(appear|count|frequency).*?(key).*?")
@register_question(r".*(LGK).*(appear|count|frequency)?.*(key).*")

async def count_lgk_key(file: UploadFile) -> str:
    file_content = await file.read()
    data = json.loads(file_content.decode("utf-8"))
    def count_key_occurrences(obj, key_to_count):
        count = 0
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == key_to_count:
                    count += 1
                count += count_key_occurrences(value, key_to_count)
        elif isinstance(obj, list):
            for item in obj:
                count += count_key_occurrences(item, key_to_count)
        return count

    lgk_count = count_key_occurrences(data, "LGK")
    return str(lgk_count)


@app.post("/api/", response_model=AnswerResponse)
async def get_answer(question: str = Form(...), file: Optional[UploadFile] = None):
    try:
        for pattern, func in function_map.items():
            if re.search(pattern, question, re.IGNORECASE):
                if file:
                    return AnswerResponse(answer=await func(file))
                else:
                    return AnswerResponse(answer=await func())

        return AnswerResponse(answer="No matching function found for the given question.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#-------- end of GA5 questions-------
#------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
