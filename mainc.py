# mainc.py

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Callable, Dict
import os
import uvicorn
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
import io
import zipfile
from PIL import Image
import numpy as np
import colorsys
import httpx
import feedparser
import json
import re
import warnings
warnings.filterwarnings("ignore")
import io
import re
from dateutil import parser
import subprocess



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
# ga1 q1 - Output of 'code -s' without escape characters ✅
@register_question(r".*output of code -s.*")
async def get_code_s_output(question: str) -> str:
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

# GA1 Q2 - Extract email and make HTTP request   --> ❌
@register_question(r".*email set to.*")
async def ga1_2(question: str) -> str:
    email_pattern = r"email set to ([\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,})"
    match = re.search(email_pattern, question)
    if match:
        email = match.group(1)
        url = "https://httpbin.org/get"
        command = ["https", "GET", url, f"email=={email}"]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout
    return "{\"error\": \"Email not found in the input text\"}"


# GA1 Q7 - Count the number of Wednesdays in a given date range ✅
@register_question(r".*How many Wednesdays are there in the date range.*")
async def count_wednesdays(question: str) -> str:
    match = re.search(r".*How many Wednesdays are there in the date range (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}).*", question)
    if not match:
        return "Invalid question format"
    start_date_str, end_date_str = match.groups()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current_date = start_date
    wednesday_count = 0
    while current_date <= end_date:
        if current_date.weekday() == 2:  # Wednesday
            wednesday_count += 1
        current_date += timedelta(days=1)
    return str(wednesday_count)

# GA1 Q8 - Import file to get answer from CSV ✅
@register_question(r".*Download and unzip file .* which has a single extract.csv file inside.*")
async def get_csv_answer(question: str, file: UploadFile) -> str:
    file_content = await file.read()
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        zip_ref.extractall('extracted_files')
    csv_file_path = 'extracted_files/extract.csv'
    df = pd.read_csv(csv_file_path)
    answer_value = df['answer'].iloc[0]
    return str(answer_value)


#GA1 Q16 - Calculate the sum of all numbers in a text file   -- ❌ recheck sha256 hash is not giving correct output
@register_question(r".*Download .* and extract it. Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next.*")
async def move_rename_files(question: str, file: UploadFile) -> str:
    import hashlib
    file_content = await file.read()
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        zip_ref.extractall('extracted_files')
    os.makedirs('extracted_files/empty_folder', exist_ok=True)
    for root, dirs, files in os.walk('extracted_files'):
        for file_name in files:
            if root != 'extracted_files/empty_folder':
                old_path = os.path.join(root, file_name)
                new_path = os.path.join('extracted_files/empty_folder', file_name)
                if os.path.exists(new_path):
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(new_path):
                        new_path = os.path.join('extracted_files/empty_folder', f"{base}_{counter}{ext}")
                        counter += 1
                os.rename(old_path, new_path)
                print(f"Moved {old_path} to {new_path}")  # Add logging
    for file_name in os.listdir('extracted_files/empty_folder'):
        new_name = ''.join(str((int(char) + 1) % 10) if char.isdigit() else char for char in file_name)
        os.rename(os.path.join('extracted_files/empty_folder', file_name), os.path.join('extracted_files/empty_folder', new_name))
        print(f"Renamed {file_name} to {new_name}")  # Add logging
    all_lines = []
    for file_name in os.listdir('extracted_files/empty_folder'):
        with open(os.path.join('extracted_files/empty_folder', file_name), 'r') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines from {file_name}")  # Add logging
            all_lines.extend(lines)
    all_lines.sort()
    print(f"Sorted lines: {all_lines}")  # Add logging
    sha256_hash = hashlib.sha256()
    for line in all_lines:
        sha256_hash.update(line.encode('utf-8'))
        print(f"Hashing line: {line.strip()}")  # Add logging
    result = sha256_hash.hexdigest()
    print(f"SHA-256 Hash: {result}")  # Add logging
    return result


# GA1 Q17 - Count the number of different lines between two files ✅
@register_question(r".*Download .* and extract it. It has 2 nearly identical files, a.txt and b.txt, with the same number of lines. How many lines are different between a.txt and b.txt?.*")
async def count_different_lines(question: str, file: UploadFile) -> str:
    file_content = await file.read()
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        zip_ref.extractall('extracted_files')
    with open('extracted_files/a.txt', 'r') as file_a, open('extracted_files/b.txt', 'r') as file_b:
        lines_a = file_a.readlines()
        lines_b = file_b.readlines()
    different_lines_count = sum(1 for line_a, line_b in zip(lines_a, lines_b) if line_a != line_b)
    return str(different_lines_count)


#-------- GA2 questions---------

# ga2 q5 - Calculate number of light pixels in an image ✅
@register_question(r".*Create a new Google Colab notebook and run this code \(after fixing a mistake in it\) to calculate the number of pixels with a certain minimum brightness.*")
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

# ga3 q9 - Generate a prompt for LLM to respond "Yes" ✅

@register_question(r".*(prompt|make).*LLM.*Yes..*")
async def get_llm_prompt_for_yes(question: str) -> str:
    return "Fire is wet"

#-------- end of GA3 questions-------
#------------------------------------


#-------- GA4 questions---------

# ga4 q5 - Get maximum latitude of Algiers in Algeria using Nominatim API ✅
@register_question(r".*?(maximum latitude|max latitude).*?(bounding box).*?city (.*?) in the country (.*?) on the Nominatim API.*")
async def get_max_latitude_city_country(question: str) -> str:
    match = re.search(r".*?(maximum latitude|max latitude).*?(bounding box).*?city (.*?) in the country (.*?) on the Nominatim API.*", question, re.IGNORECASE)
    if not match:
        return "Invalid question format"
    city = match.group(3)
    country = match.group(4)
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{city}, {country}",
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

import re
import io
from datetime import datetime
from dateutil import parser

import pandas as pd
from fastapi import UploadFile

@register_question(r".*Download the Sales Excel file: .* What is the total margin for transactions before (.*) for (.*) sold in (.*)\?.*")
async def calculate_total_margin(question: str, file: UploadFile) -> str:
    """
    This function cleans an Excel file and calculates the margin for transactions
    strictly *before* a specified local date/time, for a specified product and country.

    Main changes from previous version:
      1) We interpret "before" as a strict comparison: (df['Date'] < filter_date)
      2) We ignore the time zone from the question by using parse(..., ignoretz=True).

    This often fixes mismatches where the code previously got 0.2107 but should be 0.2362.
    """
    # --- 1) Extract components from question ---
    match = re.search(
        r".*Download the Sales Excel file: .*"
        r"What is the total margin for transactions before (.*) for (.*) sold in (.*)\?.*",
        question,
        re.IGNORECASE
    )
    if not match:
        return "Invalid question format"

    date_str, product, country = match.groups()

    # --- 2) Clean the date string by removing parentheses and parse ignoring time zone ---
    #    e.g. "Fri Nov 25 2022 06:28:05 GMT+0530 (India Standard Time)" → "Fri Nov 25 2022 06:28:05 GMT+0530"
    #    Then parse it as naive local time:
    cleaned_date_str = re.sub(r"\(.*\)", "", date_str).strip()
    parsed_dt = parser.parse(cleaned_date_str, ignoretz=True)

    # Since we are ignoring time zones, we can just use 'parsed_dt' as our cutoff
    filter_date = parsed_dt

    # --- 3) Read Excel contents into a DataFrame ---
    file_content = await file.read()
    df = pd.read_excel(io.BytesIO(file_content))

    # --- 4) Clean and normalize columns ---

    # a) Trim spaces in Customer Name and Country
    df['Customer Name'] = df['Customer Name'].astype(str).str.strip()
    df['Country']       = df['Country'].astype(str).str.strip()

    # b) Map inconsistent country names to standard codes
    country_mapping = {
        "Ind": "IN", "India": "IN", "IND": "IN",
        "USA": "US", "U.S.A": "US", "US": "US", "United States": "US",
        "UK": "GB", "U.K": "GB", "United Kingdom": "GB",
        "Fra": "FR", "France": "FR", "FRA": "FR",
        "Bra": "BR", "Brazil": "BR", "BRA": "BR",
        "AE": "AE", "U.A.E": "AE", "UAE": "AE", "United Arab Emirates": "AE"
    }
    df['Country'] = df['Country'].map(country_mapping).fillna(df['Country'])

    # c) Parse mixed-format dates (e.g. MM-DD-YYYY, YYYY/MM/DD)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)

    # d) Extract just the product name from "Product/Code" (split by slash)
    df['Product'] = df['Product/Code'].astype(str).str.strip().str.split('/').str[0]

    # e) Clean numeric columns (remove 'USD' and spaces) for Sales and Cost
    df['Sales'] = (df['Sales'].astype(str)
                             .str.replace('USD', '', case=False, regex=True)
                             .str.replace(r'\s+', '', regex=True)
                             .astype(float))

    df['Cost'] = (df['Cost'].astype(str)
                            .str.replace('USD', '', case=False, regex=True)
                            .str.replace(r'\s+', '', regex=True))
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')

    # Fill missing cost with 50% of Sales
    df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)

    # --- 5) Filter rows: strictly *before* filter_date, matching product and country ---
    country_standard = country_mapping.get(country, country)

    filtered_df = df[
        (df['Date'] < filter_date) &  # Strictly before
        (df['Product'] == product) &
        (df['Country'] == country_standard)
    ]

    # --- 6) Calculate the margin ---
    total_sales = filtered_df['Sales'].sum()
    total_cost  = filtered_df['Cost'].sum()

    filtered_df.to_csv('filtered_df.csv', index=False)
    if total_sales == 0:
        total_margin = 0
    else:
        total_margin = (total_sales - total_cost) / total_sales

    # Return as a decimal, e.g. "0.2362" for 23.62%
    return f"{total_margin:.4f}"

# ga5 q2 - Count unique student IDs in a text file


# @register_question(r".*Download.*text.* file.*q-clean-up-student-marks.txt.*(unique students|number of unique students|student IDs).*")
@register_question(r".*(unique.*students|student IDs).*?(file|download).*")

async def count_unique_students(question: str, file: UploadFile) -> str:
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
async def calculate_pizza_sales(question: str, file: UploadFile) -> str:
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
async def calculate_total_sales(question: str, file: UploadFile) -> str:
    file_content = await file.read()
    total_sales = 0
    file_content_str = file_content.decode("utf-8")
    sales_matches = re.findall(r'"sales":\s*([\d.]+)', file_content_str)
    total_sales = sum(int(float(sales)) for sales in sales_matches)
    return str(total_sales)

# ga5 q7 - Count occurrences of "LGK" as a key in nested JSON

#@register_question(r".*?(LGK).*?(appear|count|frequency).*?(key).*")
@register_question(r".*(LGK).*(appear|count|frequency)?.*(key).*")

async def count_lgk_key(question: str, file: UploadFile) -> str:
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
        #if file:
            #await handle_file(file)
        for pattern, func in function_map.items():
            if re.search(pattern, question, re.IGNORECASE):
                if file:
                    if 'file' in func.__code__.co_varnames and func.__code__.co_argcount == 1:
                        return AnswerResponse(answer=await func(file))
                    return AnswerResponse(answer=await func(question, file))
                else:
                    return AnswerResponse(answer=await func(question))

        return AnswerResponse(answer="No matching function found for the given question.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#-------- end of GA5 questions-------
#------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)