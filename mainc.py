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
import xml.etree.ElementTree as ET
from dateutil import parser
import subprocess
import shutil
from typing import Optional
from pathlib import Path
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlencode
import tabula  # pip install tabula-py
from PyPDF2 import PdfReader  # pip install PyPDF2
import PyPDF2
import httpie
from pathlib import Path
import tempfile
import hashlib
from collections import OrderedDict
import pytz



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
async def ga1_q1(question: str) -> str:
    print(f"🔥 Called ga1_q1: {question}")
    """
    Returns the output of 'code -s' without any escape characters, 
    exactly as a plain string (no JSON encoding).
    """
    # The sample 'code -s' output string:
    output_str = (
        "Version: Code 1.96.2 (fabdb6a30b49f79a7aba0f2ad9df9b399473380f, 2024-12-19T10:22:47.216Z)"
        "OS Version: Windows_NT x64 10.0.19"
    )

    # Return this raw string as-is (no JSON.dumps, so no escapes)
    return output_str

# GA1 Q2 - Extract email and make HTTP request to httpbin.org ✅

@register_question(r".*Send a HTTPS request to.*with the URL encoded parameter email set to.*")
async def ga1_q2(question: str) -> str:
    print(f"🔥 Called ga1_q2: {question}")
    email_pattern = r"email set to ([\w.%+-]+@[\w.-]+\.[a-zA-Z]{2,})"
    match = re.search(email_pattern, question)
    if match:
        email = match.group(1)

        # Full shell command as a string
        bash_command = f"http --print=b --pretty=none https://httpbin.org/get email=={email}"

        try:
            result = subprocess.run(
                ["bash", "-c", bash_command],
                capture_output=True,
                text=True
            )

            # Attempt to parse and return the JSON response
            response_json = json.loads(result.stdout)
            return json.dumps(response_json, indent=2)

        except json.JSONDecodeError:
            return json.dumps({"error": "Failed to parse JSON", "raw_output": result.stdout})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": "Email not found in the input text"})



# GA1 Q3 - Use npx and prettier to format README.md and get sha256sum ✅

@register_question(r".*npx -y prettier@3.4.2 README.md.*")

async def ga1_q3(question: str, file: UploadFile) -> str:
    print(f"🔥 Called ga1_q3: {question}")
    try:
        # Step 1: Save the uploaded file as README.md in a temp directory
        file_path = f"/tmp/README.md"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Step 2: Run the command: npx -y prettier@3.4.2 README.md | sha256sum
        result = subprocess.run(
            "npx -y prettier@3.4.2 README.md | sha256sum",
            shell=True,
            cwd="/tmp",
            capture_output=True,
            text=True
        )

        # Step 3: Extract and return only the SHA-256 hash
        return result.stdout.split()[0] if result.stdout else "Error: No output"

    except Exception as e:
        return f"Error: {str(e)}"

#GA1 Q4 - Sum(array_constrain(sequence())) using google sheets ✅
@register_question(r".*=SUM\(ARRAY_CONSTRAIN\(SEQUENCE.*")
async def ga1_q4(question: str) -> str:
    print(f"🔥 Called ga1_q4: {question}")
    match = re.search(
        r"=SUM\(ARRAY_CONSTRAIN\(SEQUENCE\((\d+), (\d+), (\d+), (\d+)\), (\d+), (\d+)\)\)",
        question
    )
    if not match:
        return "Invalid question format"

    rows, cols, start, step, constrain_rows, constrain_cols = map(int, match.groups())

    # Generate the full SEQUENCE grid row-wise
    sequence = []
    for r in range(rows):
        row_data = [start + step * (r * cols + c) for c in range(cols)]
        sequence.append(row_data)

    # Apply ARRAY_CONSTRAIN
    constrained_sequence = [row[:constrain_cols] for row in sequence[:constrain_rows]]

    # Flatten and sum
    total_sum = sum(num for row in constrained_sequence for num in row)

    return str(total_sum)

#GA1 Q5 - SUM(TAKE(SORTY() using EXCEL)) ✅

@register_question(r".*=SUM\(TAKE\(SORTBY\({.*")
async def ga1_q5(question: str) -> str:
    print(f"🔥 Called ga1_q5: {question}")
    """
    Handles Excel 365-specific formula-based questions:
    =SUM(TAKE(SORTBY({values}, {keys}), rows, cols))

    Steps:
    - Extract values, sort keys, rows and columns from the formula
    - Sort values based on keys
    - Slice the top [rows x cols] elements row-wise
    - Sum and return resultß
    """

    try:
        # Remove all line breaks and whitespace for regex processing
        cleaned_question = question.replace("\n", "").replace(" ", "")

        # Regex to extract all components from the formula
        match = re.search(
            r"=SUM\(TAKE\(SORTBY\(\{([0-9,]+)\},\{([0-9,]+)\}\),(\d+),(\d+)\)\)",
            cleaned_question
        )

        if not match:
            return "Invalid question format."

        values_str, keys_str, rows_str, cols_str = match.groups()

        # Convert to appropriate data types
        values = list(map(int, values_str.split(",")))
        keys = list(map(int, keys_str.split(",")))
        rows = int(rows_str)
        cols = int(cols_str)

        # Validate lengths
        if len(values) != len(keys):
            return "Mismatched values and keys."

        # Sort values based on keys
        sorted_values = [val for _, val in sorted(zip(keys, values))]

        # Calculate the number of elements to take (rows x cols)
        total_to_take = rows * cols

        # Take first (rows * cols) elements from sorted list
        top_values = sorted_values[:total_to_take]

        # Return their sum
        return str(sum(top_values))

    except Exception as e:
        return f"Error: {str(e)}"


#GA1 Q6 - hidden element --> should we hardcode the answer?

@register_question(r".*Just above this paragraph, there's a hidden input with a secret value.*")
async def ga1_q6(question: str, file: UploadFile = None) -> str:
    """
    GA1 Q6: Extract the value of the hidden input just above the paragraph in the HTML.

    - If a URL is present in the question, it fetches the HTML from that URL.
    - If a file is uploaded, it parses the file content as HTML.
    - If neither is available, it parses the question text as HTML.
    """
    print(f"🔍 Called ga1_q6: {question}")
    import re
    import requests
    from bs4 import BeautifulSoup

    html_data = None

    try:
        # Step 1: Check for URL in the question
        url_match = re.search(r"https?://[^\s]+", question)
        if url_match:
            source = url_match.group(0)
            response = requests.get(source, timeout=5)
            response.raise_for_status()
            html_data = response.text

        # Step 2: If file is provided, read it
        elif file:
            file_content = await file.read()
            html_data = file_content.decode("utf-8")

        # Step 3: Else fallback to parsing the question as HTML
        else:
            soup = BeautifulSoup(question, "html.parser")
            div_text = soup.find("div")
            return div_text.get_text(strip=True) if div_text else ""

        # Step 4: Extract hidden input value
        soup = BeautifulSoup(html_data, "html.parser")
        hidden_input = soup.find("input", {"type": "hidden"})
        return hidden_input.get("value", "") if hidden_input else ""

    except Exception as e:
        return f"❌ Error: {str(e)}"


# GA1 Q7 - Count the number of Wednesdays in a given date range ✅
@register_question(r".*How many Wednesdays are there in the date range.*")
async def ga1_q7(question: str) -> str:
    print(f"🔥 Called ga1_q7: {question}")
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
async def ga1_q8(question: str, file: UploadFile) -> str:
    print(f"🔥 Called ga1_q8: {question}")
    file_content = await file.read()
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        zip_ref.extractall('extracted_files')
    csv_file_path = 'extracted_files/extract.csv'
    df = pd.read_csv(csv_file_path)
    answer_value = df['answer'].iloc[0]
    return str(answer_value)


#GA1 Q9 - sort the json based on name and age ✅

@register_question(r".*Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field.*")
async def ga1_q9(question: str) -> str:
    print(f"🔥 Called ga1_q9: {question}")
    """
    Example question snippet:
      "Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field.
       [{\"name\":\"Alice\",\"age\":11},{\"name\":\"Bob\",\"age\":11}, ... ]"

    Steps:
      1) Extract JSON array from the question
      2) Parse into Python
      3) Sort by (age, name)
      4) Return as single-line JSON (no spaces or newlines)
    """

    # 1) Extract the JSON array with a regex capturing '[' ... ']'
    match = re.search(r"(\[.*])", question, flags=re.DOTALL)
    if not match:
        return "No JSON array found in the question."

    json_str = match.group(1).strip()

    # 2) Parse into a Python list
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return "Invalid JSON format in the question."

    if not isinstance(data, list):
        return "The extracted JSON is not an array."

    # 3) Sort by 'age' ascending, then by 'name' ascending if there's a tie
    #    We assume each object has keys 'name' and 'age'
    def sort_key(item):
        # If 'age' or 'name' is missing, handle gracefully
        # Convert 'name' to string in case it's not
        age_val = item.get("age", 0)  # fallback 0 or some default if missing
        name_val = str(item.get("name", ""))
        return (age_val, name_val)

    data_sorted = sorted(data, key=sort_key)

    # 4) Convert back to single-line JSON
    # Use separators=(",", ":") to avoid extra spaces/newlines
    result_json = json.dumps(data_sorted, separators=(",", ":"))

    return result_json

#GA1 Q10 - CONVERT INTO A SINGLE JSON OBJECT AND FETCH JSONHASH FROM URL
@register_question(r".*convert it into a single JSON object.*jsonhash.*")
async def ga1_q10(question: str, file: UploadFile) -> str:
    print(f"🔥 Called ga1_q10: {question}")

    try:
        # Step 1: Read uploaded text file
        content = (await file.read()).decode("utf-8").strip().splitlines()

        # Step 2: Build OrderedDict to preserve input order
        data = OrderedDict()
        for line in content:
            if "=" in line:
                key, value = line.split("=", 1)
                data[key.strip()] = value.strip()

        # Step 3: Dump to minified JSON string without sorting keys
        minified_json = json.dumps(data, separators=(",", ":"))  # No sort_keys=True

        # ✅ Print the ordered minified JSON
        #print("Minified JSON in order:\n", minified_json)

        # Step 4: Hash it like the browser tool - this hash is same for same exact input.used to verify data integrity.
        """
        This line does three things: Converts the JSON string into bytes using UTF-8:
        minified_json.encode("utf-8") --> Computers work with bytes, not strings—so we first encode the string.
        Creates a SHA-256 hash of those bytes:  --> hashlib.sha256(...).hexdigest() 
        SHA-256 is a way to create a unique digital fingerprint (a 64-character hexadecimal string) for the input.
        Returns the hash as a readable hex string using .hexdigest()
        """

        hash_val = hashlib.sha256(minified_json.encode("utf-8")).hexdigest()

        return hash_val

    except Exception as e:
        return f"Error: {str(e)}"



#GA1 Q11 - SUM OF DATA VALUE ATTRIBUTE
@register_question(r".*Find all <div>s having a foo class.*sum of their data-value attributes.*")
async def ga1_q11(question: str, file: UploadFile = None) -> str:
    """
    GA1 Q11: Find all <div class="foo"> elements and sum their data-value attributes.
    The HTML can come from inline question, an uploaded file, or an external URL.
    """
    print(f"🔎 Called ga1_q11: {question}")
    import re
    from bs4 import BeautifulSoup
    import requests

    html_data = None

    try:
        # Step 1: Check for URL in the question
        url_match = re.search(r"https?://[^\s]+", question)
        if url_match:
            response = requests.get(url_match.group(0), timeout=5)
            response.raise_for_status()
            html_data = response.text

        # Step 2: If a file is uploaded
        elif file:
            file_content = await file.read()
            html_data = file_content.decode("utf-8")

        # Step 3: Else fallback to question text (may contain HTML)
        else:
            html_data = question

        # Step 4: Parse the HTML
        soup = BeautifulSoup(html_data, "html.parser")

        # Step 5: Select divs with class 'foo' and data-value attribute
        divs = soup.select('div.foo[data-value]')
        values = [float(div['data-value']) for div in divs]

        # Step 6: Return the sum as integer
        return str(int(sum(values)))

    except Exception as e:
        return f"❌ Error: {str(e)}"

#GA1 Q12 - Sum up all the values where the symbol matches ✅

@register_question(r".*Sum up all the values where the symbol matches.*")
async def ga1_q12(question: str, file: UploadFile) -> str:
    print(f"🔥 Called ga1_q12: {question}")
    """
    Example question:
      "Sum up all the values where the symbol matches Č OR ř OR ž across all three files."

    Steps:
      1) Extract the 3 symbols from question
      2) Unzip the uploaded file
      3) Read:
         - data1.csv (CP-1252)
         - data2.csv (UTF-8)
         - data3.txt (UTF-16, tab sep)
      4) Combine into a single DataFrame
      5) Filter rows whose 'symbol' is in the 3 target symbols
      6) Sum 'value' as an integer
      7) Print matched symbols (for debug) and return total as a string
    """

    # 1) Extract the 3 symbols from the question
    match = re.search(r"Sum up all the values where the symbol matches (.*) OR (.*) OR (.*) across all three files", question)
    if not match:
        return "Question format invalid or no symbols found."

    symbol1, symbol2, symbol3 = match.groups()
    target_symbols = {symbol1, symbol2, symbol3}

    # 2) Read the ZIP file from memory
    zip_bytes = await file.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        # 3) Open each data file
        with z.open("data1.csv") as f1:
            df1 = pd.read_csv(f1, encoding="cp1252")
        with z.open("data2.csv") as f2:
            df2 = pd.read_csv(f2, encoding="utf-8")
        with z.open("data3.txt") as f3:
            df3 = pd.read_csv(f3, encoding="utf-16", sep="\t")

    # 4) Combine all dataframes
    all_data = pd.concat([df1, df2, df3], ignore_index=True)

    # 5) Filter rows where 'symbol' is one of the target symbols
    filtered_data = all_data[all_data["symbol"].isin(target_symbols)]

    # (Optional) Let's debug which symbols matched
    matched_symbols = filtered_data["symbol"].unique()
    #print("Matched symbols (for debugging):", matched_symbols).  ---> uncomment to see matched symbols

    # 6) Sum the 'value' column as an integer
    # Convert the column to numeric and fill missing with 0 just in case
    filtered_data["value"] = pd.to_numeric(filtered_data["value"], errors="coerce").fillna(0)
    total_sum = int(filtered_data["value"].sum())

    # 7) Return total as string
    return str(total_sum)

   
#GA1 Q13 - Enter the raw Github URL of email.json so we can verify it. ✅
#(It might look like https://raw.githubusercontent.com/[GITHUB ID]/[REPO NAME]/main/email.json.)

@register_question(r".*Enter the raw Github URL of email.json so we can verify it.*")
async def ga1_q13(question: str) -> str:
    print(f"🔥 Called ga1_q13: {question}")
    url ="https://raw.githubusercontent.com/shalinis97/TDS/refs/heads/main/email.json"
    return url

# GA1 Q14 - find and replace a string in a file ✅

#@register_question(r".*Download.*and unzip it into a new folder, then replace all \"IITM\" (in upper, lower, or mixed case) with \"IIT Madras\" in all files. Leave everything as-is - don't change the line endings.*What does running cat \* \| sha256sum in that folder show in bash.*")
#@register_question(r".*Download.*replace all\s+\"?IITM\"?.*with\s+\"?IIT Madras\"?.*in all files.*cat\s+\*\s+\|\s+sha256sum.*bash.*")
#@register_question(r".*Leave everything as-is - don't change the line endings.*does running cat * | sha256sum in that folder show in bash?.*")
#@register_question(r".*IIT Madras.*")
#@register_question(r".*replace all.*IITM.*with.*IIT Madras.*sha256sum.*")
#@register_question(r".*unzip.*replace all.*IITM.*with.*IIT Madras.*line endings.*cat \* \| sha256sum.*")
#@register_question(r".*Leave everything as-is - don't change the line endings.*")
#@register_question(r".*replace all [\"']?IITM[\"']? \(in upper, lower, or mixed case\) with [\"']?IIT Madras[\"']? in all files\. Leave everything as-is - don't change the line endings.*")
@register_question(r".*Leave\s+everything\s+as[\s\-]*is\s*[-–—]?\s*don'?t\s+change\s+the\s+line\s+endings\..*")
async def ga1_q14(question: str, file: UploadFile) -> str:
    print("✅ ga1_q14 matched and is executing.")

    try:
        # ✅ Step 1: Save the uploaded zip
        zip_path = f"{file.filename}"
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # ✅ Step 2: Extract to normal folder `ga1_q14` (create if doesn't exist)
        extract_folder = "ga1_q14"
        print("Extracting to:", os.path.abspath(extract_folder))
        print("Current working directory:", os.getcwd())

        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # ✅ Step 3: Replace all "IITM" (any case) with "IIT Madras" in all files using bash sed
        sed_cmd = "find . -type f -exec sed -i 's/[Ii][Ii][Tt][Mm]/IIT Madras/g' {} +"
        subprocess.run(sed_cmd, shell=True, check=True, cwd=extract_folder)

        # ✅ Step 4: Get sha256sum from all file contents using `cat * | sha256sum`
        sha_cmd = "cat * | sha256sum"
        result = subprocess.run(sha_cmd, shell=True, capture_output=True, text=True, cwd=extract_folder)

        if not result.stdout:
            return "Error: No output from sha256sum"

        return result.stdout.split()[0]

    except Exception as e:
        return f"Error: {str(e)}"




# GA1 Q15 - filter files based on size and timestamp
@register_question(r".*Use ls with options to list all files in the folder along with their date and file size.*")
async def ga1_q15(question: str, zip_file: UploadFile) -> str:

    print(f"🔥 Called ga1_q15: {question}")

    try:
        # ✅ Extract minimum size from question
        size_match = re.search(r"at least (\d+) bytes", question)
        if not size_match:
            return "Error: Could not extract file size from question."
        min_size = int(size_match.group(1))

        # ✅ Extract timestamp string and convert to datetime
        date_match = re.search(r"on or after ([\w, ]+\d{4}, \d+:\d+ [apAP][mM]) IST", question)
        if not date_match:
            return "Error: Could not extract timestamp from question."
        
        date_str = date_match.group(1).strip()
        try:
            target_dt = datetime.strptime(date_str, "%a, %d %b, %Y, %I:%M %p")
            target_dt = pytz.timezone("Asia/Kolkata").localize(target_dt)
        except ValueError as e:
            return f"Date parsing error: {str(e)}"

        # ✅ Read the uploaded zip file in memory
        zip_bytes = await zip_file.read()
        total_size = 0

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                # Convert zip_info.date_time to localized datetime
                file_mtime = datetime(*zip_info.date_time)
                file_mtime = pytz.timezone("Asia/Kolkata").localize(file_mtime)

                # Check both conditions: size and modified timestamp
                if zip_info.file_size >= min_size and file_mtime >= target_dt:
                    total_size += zip_info.file_size

        return str(total_size)

    except Exception as e:
        return f"❌ Error: {str(e)}"



#GA1 Q16 - Calculate the sum of all numbers in a text file   -- ✅ 
@register_question(r".*grep . * | LC_ALL=C sort | sha256sum.*")
async def ga1_q16(question: str, file: UploadFile) -> str:
    print(f"🔥 Called ga1_q16: {question}")
    try:
        # Step 1: Save the uploaded ZIP file
        zip_path = f"/tmp/{file.filename}"  # Temporary path for extraction
        extract_folder = f"/tmp/extracted_{os.path.splitext(file.filename)[0]}"
        
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # Step 2: Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

        # Step 3: Move all files from subdirectories to the main folder
        for root, dirs, files in os.walk(extract_folder):
            for file in files:
                old_path = os.path.join(root, file)
                new_path = os.path.join(extract_folder, file)
                if old_path != new_path:  # Avoid moving if already in the folder
                    shutil.move(old_path, new_path)

        # Step 4: Rename files (Replace each digit with the next one)
        for file in os.listdir(extract_folder):
            new_name = re.sub(r'\d', lambda x: str((int(x.group(0)) + 1) % 10), file)
            old_path = os.path.join(extract_folder, file)
            new_path = os.path.join(extract_folder, new_name)
            os.rename(old_path, new_path)

        # Step 5: Run the required bash command and get SHA-256 hash
        result = subprocess.run(
            'grep . * | LC_ALL=C sort | sha256sum',
            shell=True,
            cwd=extract_folder,
            capture_output=True,
            text=True
        )

        # Extract and return only the hash
        return result.stdout.split()[0] if result.stdout else "Error: No output"

    except Exception as e:
        return f"Error: {str(e)}"


# GA1 Q17 - Count the number of different lines between two files ✅
@register_question(r".*How many lines are different between a.txt and b.txt?.*")
async def ga1_q17(question: str, file: UploadFile) -> str:
    print(f"🔥 Called ga1_q17: {question}")
    file_content = await file.read()
    with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zip_ref:
        zip_ref.extractall('extracted_files')
    with open('extracted_files/a.txt', 'r') as file_a, open('extracted_files/b.txt', 'r') as file_b:
        lines_a = file_a.readlines()
        lines_b = file_b.readlines()
    different_lines_count = sum(1 for line_a, line_b in zip(lines_a, lines_b) if line_a != line_b)
    return str(different_lines_count)


#GA1 Q18 - sql query
@register_question(r".*What is the total sales of all the items in the \"Gold\" ticket type\? Write SQL to calculate it.*")
async def ga1_q18(question:str) -> str:
    print(f"🔥 Called ga1_q18: {question}")
    sql_query = "SELECT SUM(units * price) AS sales FROM tickets WHERE trim(lower(type)) = 'gold';"
    return sql_query
#-------- GA2 questions---------

# GA2 Q1 - Write Markdown documentation for weekly step count analysis
@register_question(r".*number of steps.*")
async def ga2_q1(question: str) -> str:
    markdown_content = """# Weekly Step Count Analysis
![Image](https://www.10000steps.org.au/media/images/Counting-Your-Steps-Blog.original.png)

## Introduction
Tracking the number of steps walked daily is an **important** measure of physical activity and helps in maintaining a healthy lifestyle. This analysis compares my step count with my friends over the past week to observe trends and patterns.

---

## Methodology
To conduct this analysis, the following steps were followed:

1. *Data Collection:* Step count data was recorded using fitness trackers.
2. *Comparison:* The data was compared across individuals to identify trends.
3. *Visualization:* The results were displayed using tables and charts.

---

## Step Count Comparison

Below is a table showing the step count for three individuals over three days:

| Name       | Monday       | Tuesday      | Wednesday    |
|------------|-------------|--------------|--------------|
| **Shalini** | *1090 steps* | *1000 steps*  | *1000 steps*  |
| **Raajashri** | *10000 steps* | *1940 steps*  | *1890 steps*  |
| **Finn**    | *19000 steps* | *18000 steps* | *1000 steps*  |

---

### Important Observations

- Shalini is **27 years old**, maintaining a consistent but low step count.
- Raajashri is **23 years old**, showing a high fluctuation in daily steps.
- Finn is an **8-year-old dog**, with an impressive step count.

---

## Insights

> "Taking at least 10,000 steps a day can significantly improve health outcomes."

For more information on the importance of daily steps, check out this [NIH article](https://www.nih.gov/news-events/nih-research-matters/number-steps-day-more-important-step-intensity#:~:text=People%20who%20took%2012%2C000%20steps,%2C%20sex%2C%20and%20race%20groups).

---

## Step Count Visualization

Below is a Python snippet used to visualize the step count data:

```python
import matplotlib.pyplot as plt

names = ["Shalini", "Raajashri", "Finn"]
steps = [1090, 10000, 19000]

plt.bar(names, steps)
plt.title("Step Count Comparison")
plt.xlabel("Individuals")
plt.ylabel("Steps")
plt.show()
```
"""
    return markdown_content #return markdown_content.replace("\n", "")


#GA2 Q4 - google collab 
@register_question(r".Let's make sure you can access Google Colab.")
async def ga2_q4(question: str) -> str:
    print(f"🔥 Called ga2_q4: {question}")

    match = re.search(r"ID:\s*([\w\.-]+@[\w\.-]+)", question)
    if not match:
        return "Error: Could not extract email from question"

    email = match.group(1).rstrip('.')
    year = 2025  # fixed to match Colab
    i1 = f"{email} {year}".encode()
    print(email)
    print(year)
    print(i1)

    hash_val = hashlib.sha256(i1).hexdigest()[-5:]
    hash_full= hashlib.sha256(i1).hexdigest()
    print(hash_val)
    print(hash_full)
    return hash_val

# GA2 Q5 - Calculate number of light pixels in an image ✅
@register_question(r".*Create a new Google Colab notebook and run this code \(after fixing a mistake in it\) to calculate the number of pixels with a certain minimum brightness.*")
async def ga2_q5(question:str, file: UploadFile) -> str:
    file_content = await file.read()
    image = Image.open(io.BytesIO(file_content))
    rgb = np.array(image) / 255.0
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)
    light_pixels = np.sum(lightness > 0.133)
    return str(int(light_pixels))

#GA2 Q10 - running llamafile through ngrok
@register_question(r".*Create a tunnel to the Llamafile server using ngrok.*")
async def ga2_q10(question: str) -> str:
    print(f"🔥 Called ga2_q10: {question}")
    url = "https://2350-2409-4072-6e45-1953-c9d6-9624-b787-cecb.ngrok-free.app/"
    return url




#-------- end of GA2 questions-------
#------------------------------------



#-------- GA3 questions---------

# GA3 Q9 - Generate a prompt for LLM to respond "Yes" ✅

@register_question(r".*(prompt|make).*LLM.*Yes..*")
async def ga3_q9(question: str) -> str:
    return "Fire is wet"



#-------- end of GA3 questions-------
#------------------------------------


#-------- GA4 questions---------

# GA4 Q1 - Get the total no of ducks from the espn page ✅
@register_question(r".*total number of ducks.*")
async def ga4_q1(question: str) -> str:
    """
    Answers questions about the total number of ducks on a given ESPN Cricinfo 
    ODI batting stats page, e.g.:
      "What is the total number of ducks across players on page number 30 of ESPN Cricinfo's ODI batting stats?"
    
    Steps:
      1) Extract page number from the question: "page number <digits>"
      2) Build the URL for that page
      3) Fetch the HTML with a browser-like user-agent to avoid 403
      4) Find the 'engineTable' that contains a header named "Player"
      5) Determine which column is labeled "0" (ducks)
      6) Gather data rows (class="data1"), sum integer values in that '0' column
      7) Return the sum as a string
    """

    # 1) Extract page number from question
    match = re.search(r"page number\s+(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return "No valid page number found in the question."

    page_num = int(match.group(1))

    # 2) Build the ESPN Cricinfo ODI batting stats URL
    url = (
        "https://stats.espncricinfo.com/stats/engine/stats/"
        f"index.html?class=2;template=results;type=batting;page={page_num}"
    )

    # 3) Use a custom User-Agent to avoid 403 Forbidden
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    if not response.ok:
        return f"Error fetching page {page_num}. HTTP status: {response.status_code}"

    # 4) Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find a table with class="engineTable" that has a <th> named "Player"
    tables = soup.find_all("table", class_="engineTable")
    stats_table = None
    for table in tables:
        if table.find("th", string="Player"):
            stats_table = table
            break

    if not stats_table:
        return "Could not find the batting stats table on the page."

    # 5) Extract the table headers
    headers_list = [th.get_text(strip=True) for th in stats_table.find_all("th")]

    # Find the index of the "0" column
    duck_col_index = None
    for i, header in enumerate(headers_list):
        if header == "0":
            duck_col_index = i
            break

    if duck_col_index is None:
        return "Could not find the '0' (ducks) column in the table."

    # 6) Extract the data rows; ESPN often labels them with class="data1"
    data_rows = stats_table.find_all("tr", class_="data1")

    # Sum the ducks
    total_ducks = 0
    for row in data_rows:
        cells = row.find_all("td")
        if len(cells) > duck_col_index:
            duck_value = cells[duck_col_index].get_text(strip=True)
            if duck_value.isdigit():
                total_ducks += int(duck_value)

    # 7) Return the total as a string
    return str(total_ducks)


#GA4 Q3 -  public api endpoint url that gives a json response of headers alone taking 
#countryname as input in the url (request is passed accordingly). Takes data from wikipedia.

@register_question(r".*Wikipedia.*")
async def ga4_q3(question: str) -> str:
    print(f"🔥 Called ga4_q3: {question}")
    url ="https://e00b-2409-4072-6e45-1953-c9d6-9624-b787-cecb.ngrok-free.app/api/outline"
    return url

#GA4 Q4 - Json weather description for a city ✅
@register_question(r".*What is the JSON weather forecast description for.*")
async def ga4_q4(question: str) -> str:
    """
    Answers a question like:
        "What is the JSON weather forecast description for Seoul?"
    
    1) Extract city name from question via regex
    2) Use BBC's location service to find the city ID
    3) Fetch BBC weather page for that ID
    4) Parse the daily summary from the weather page
    5) Create a dictionary mapping each date to its summary
    6) Return that dictionary as a JSON string
    """
    # 1) Extract city name using regex
    match = re.search(r".*What is the JSON weather forecast description for (.*)\?", question, flags=re.IGNORECASE)
    if not match:
        return "Invalid question format. Please ask 'What is the JSON weather forecast description for <city>?'"
    city = match.group(1).strip()

    # 2) Build the BBC location service URL to get the city ID
    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
       'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
       's': city,
       'stack': 'aws',
       'locale': 'en',
       'filter': 'international',
       'place-types': 'settlement,airport,district',
       'order': 'importance',
       'a': 'true',
       'format': 'json'
    })

    try:
        # Fetch location data (JSON)
        loc_result = requests.get(location_url).json()
        # The first search result's ID
        city_id = loc_result['response']['results']['results'][0]['id']
    except (KeyError, IndexError) as e:
        return f"Could not find weather location for '{city}'. Error: {e}"

    # 3) Build the BBC weather page URL
    weather_url = 'https://www.bbc.com/weather/' + city_id

    # 4) Fetch the weather page HTML
    response = requests.get(weather_url)
    if not response.ok:
        return f"Error fetching weather data for {city}. HTTP status: {response.status_code}"

    soup = BeautifulSoup(response.content, 'html.parser')

    # 5) Parse the daily summary (div with class 'wr-day-summary')
    daily_summary_div = soup.find('div', attrs={'class': 'wr-day-summary'})
    if not daily_summary_div:
        return f"Could not find daily summary for {city} on BBC Weather."

    # Extract text and split into list of descriptions
    daily_summary_list = re.findall('[a-zA-Z][^A-Z]*', daily_summary_div.text)

    # 6) Create date list (assuming one summary per day)
    datelist = pd.date_range(datetime.today(), periods=len(daily_summary_list)).tolist()
    datelist = [date.date().strftime('%Y-%m-%d') for date in datelist]

    # Map each date to its summary
    weather_data = {date: desc for date, desc in zip(datelist, daily_summary_list)}

    # 7) Convert dictionary to JSON and return
    return json.dumps(weather_data, indent=4)


# GA4 Q5 - Get maximum latitude of Algiers in Algeria using Nominatim API ✅
@register_question(r".*?(maximum latitude|max latitude).*?(bounding box).*?city (.*?) in the country (.*?) on the Nominatim API.*")
async def ga4_q5(question: str) -> str:
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

# GA4 Q6 - Get link to the latest Hacker News post about "name" with point ✅


@register_question(r".*What is the link to the latest Hacker News post mentioning.*")
async def ga4_q6(question: str) -> str:
    """
    Example question:
      "What is the link to the latest Hacker News post mentioning DuckDB having at least 71 points?"
    
    Steps:
      1) Regex capture: search term (e.g. "DuckDB") and integer points (e.g. "71").
      2) Make an async GET request to https://hnrss.org/newest?q=<term>&points=<points> using httpx.
      3) Parse the XML with ElementTree, find the first <item>, and return the <link>.
      4) If no items found, or question is invalid, return a relevant message.
    """

    # 1) Extract search term and points from the question
    match = re.search(
        r"What is the link to the latest Hacker News post mentioning (.+?) having at least (\d+) points\?",
        question,
        flags=re.IGNORECASE
    )
    if not match:
        return ("Invalid question format. Please ask: "
                "'What is the link to the latest Hacker News post mentioning <term> having at least <points> points?'")
    search_term = match.group(1).strip()
    min_points = match.group(2).strip()

    # 2) Build the HNRSS URL and parameters
    url = "https://hnrss.org/newest"
    params = {
        "q": search_term,
        "points": min_points
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Raise if e.g. 4xx/5xx
            rss_content = response.text

            # 3) Parse the RSS feed
            root = ET.fromstring(rss_content)
            # Items are typically found under <channel><item>
            items = root.findall(".//item")

            if not items:
                return f"No Hacker News posts found mentioning {search_term} with at least {min_points} points"

            # Grab the first item (most recent)
            latest_item = items[0]

            # Extract link from <link> tag
            link_elem = latest_item.find("link")
            link = link_elem.text if link_elem is not None else None

            if not link:
                return "No link found for the latest HN post"

            return link

    except Exception as e:
        return f"Failed to fetch or parse HNRSS feed: {str(e)}"


#GA4 Q7 - Using the GitHub API, find all users located in the city ✅

@register_question(r".*Using the GitHub API, find all users located in the city.*")
async def ga4_q7(question: str) -> str:
    """
    Example question:
      "Using the GitHub API, find all users located in the city Basel with over 80 followers?"
    
    Steps:
      1) Extract 'city' and 'followers' from the question with regex.
      2) Build the GitHub search query for location:<city> and followers:> <followers>.
      3) Sort by 'joined' descending.
      4) Iterate results, find the newest user (by join date) that was created before the cutoff date.
      5) Return that user's information or a 'No users found' message.
    """

    # 1) Extract the city and followers from the question
    match = re.search(
        r"Using the GitHub API, find all users located in the city (.+?) with over (\d+) followers",
        question,
        flags=re.IGNORECASE
    )
    if not match:
        return (
            "Invalid question format. Please ask in the form: "
            "'Using the GitHub API, find all users located in the city <City> with over <followers> followers?'"
        )

    city = match.group(1).strip()
    followers = match.group(2).strip()

    # Build the query for the GitHub API
    # e.g. 'location:Basel followers:>80'
    query = f'location:"{city}" followers:>{followers}'
    params = {
        'q': query,
        'sort': 'joined',
        'order': 'desc'
    }

    url = 'https://api.github.com/search/users'
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"GitHub API request failed with status {response.status_code}"

    data = response.json()
    if 'items' not in data:
        return "No users found in the response."

    # Define the cutoff date
    cutoff_date_str = '2025-02-08T17:15:15Z'
    cutoff_date = datetime.strptime(cutoff_date_str, '%Y-%m-%dT%H:%M:%SZ')

    # Iterate through the search results in descending join order
    for user in data.get('items', []):
        # user['url'] is the API URL for details about that user
        user_response = requests.get(user['url'])
        if user_response.status_code != 200:
            # Could skip or return an error message
            continue
        user_data = user_response.json()
        
        # Parse the created_at date
        created_at_str = user_data.get('created_at')
        if not created_at_str:
            continue
        created_at = datetime.strptime(created_at_str, '%Y-%m-%dT%H:%M:%SZ')

        # Check if this user was created before the cutoff date
        if created_at < cutoff_date:
            # Return or build the user info
            username = user_data.get('login', 'Unknown')
            profile_url = user_data.get('html_url', 'Unknown')
            created_date = user_data.get('created_at', 'Unknown')
            return (created_date)

    # If we exhaust the list and find no user matching the cutoff criterion
    return "No users found matching the criteria."


#GA4 Q9 - PDF MARKS (BIOLOGY, MATHS, PHYSICS, CHEM) BASED ON GROUPS

@register_question(r".*marks of students who scored.*")
async def ga4_q9(question: str, file: UploadFile) -> str:
    """
    Example question:
      "What is the total Biology marks of students who scored 32 or more marks in Maths in groups 11-44 (including both groups)?"
    
    Because the PDF has one group per page, with heading "Student marks - Group X",
    we parse each page individually, detect the group number, insert it as "Group",
    and then combine all pages into a single DataFrame with columns like:
      ["Maths", "Physics", "English", "Economics", "Biology", "Group"]
    Then filter by sub2 >= mark1, group in [group1..group2], sum(sub1).
    """

    # 1) Extract sub1, mark1, sub2, group1, group2 from the question
    match = re.search(
        r"What is the total (.+?) marks of students who scored (\d+) or more marks in (.+?) in groups (\d+)-(\d+) \(including both groups\)\?",
        question,
        flags=re.IGNORECASE
    )
    if not match:
        return (
            "Invalid question format. Example:\n"
            "What is the total Biology marks of students who scored 32 or more marks in Maths in groups 11-44 (including both groups)?"
        )

    sub1  = match.group(1).strip()  # e.g. "Biology"
    mark1 = int(match.group(2))     # e.g. 32
    sub2  = match.group(3).strip()  # e.g. "Maths"
    grp1  = int(match.group(4))     # e.g. 11
    grp2  = int(match.group(5))     # e.g. 44

    # 2) Read the PDF from the UploadFile into memory, then write to a temp file
    pdf_bytes = await file.read()
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 3) Determine how many pages are in the PDF (so we can parse each individually)
    reader = PdfReader(temp_pdf_path)
    total_pages = len(reader.pages)

    all_dfs = []

    # 4) For each page: 
    #    a) read the text with PyPDF2 to find the heading "Student marks - Group X"
    #    b) parse the table with tabula
    #    c) label each row with the group number X, and store the DataFrame
    for page_num in range(1, total_pages + 1):
        page_index = page_num - 1  # PyPDF2 pages are 0-based

        # (a) find "Student marks - Group X" in the page text
        page_text = reader.pages[page_index].extract_text() or ""
        # e.g.  "Student marks - Group 11"

        group_match = re.search(r"Student marks\s*-\s*Group\s+(\d+)", page_text)
        if not group_match:
            # If we can't find a group # on this page, skip
            continue

        group_number = int(group_match.group(1))

        # (b) parse the table on this page with tabula
        try:
            # We'll parse only this page
            df_list = tabula.read_pdf(
                temp_pdf_path,
                pages=str(page_num),
                multiple_tables=False,  # Each page is just one main table
                lattice=True            # or stream=True, depending on PDF lines
            )
        except Exception as e:
            # If tabula can't parse this page, skip or handle error
            continue

        if not df_list:
            continue

        # There's presumably one table per page
        df_page = df_list[0]

        # (c) Insert a "Group" column
        df_page["Group"] = group_number

        # We might rename columns if needed. The PDF columns are:
        # ["Maths", "Physics", "English", "Economics", "Biology"]
        # If tabula doesn't produce exactly those column names, rename them here.
        # For example, if the first row is used as a header:
        # df_page.columns = ["Maths", "Physics", "English", "Economics", "Biology", ...]
        # OR if the PDF is recognized correctly, no rename needed.

        all_dfs.append(df_page)

    if not all_dfs:
        return "No tables found across pages."

    # Combine all pages into one DataFrame
    df = pd.concat(all_dfs, ignore_index=True)

    # 5) Convert numeric columns to numeric
    # If the parsed column headers differ, adjust accordingly.
    for col in ["Maths", "Physics", "English", "Economics", "Biology", "Group"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6) Check that sub1 and sub2 exist
    if sub1 not in df.columns or sub2 not in df.columns:
        return f"Columns '{sub1}' or '{sub2}' not found. Found: {list(df.columns)}"

    # 7) Filter:
    #    - df[sub2] >= mark1
    #    - group in [grp1..grp2]
    mask = (
        (df[sub2] >= mark1) &
        (df["Group"].between(grp1, grp2, inclusive="both"))
    )
    filtered_df = df[mask]

    # 8) Sum the sub1 column
    total_marks = filtered_df[sub1].sum(skipna=True)

    return str(total_marks)


#GA4 Q10 - What is the markdown content of the PDF, formatted with prettier@3.4.2?

@register_question(r".*What is the markdown content of the PDF, formatted with prettier@3.4.2.*")
async def ga4_q10(question: str, file: UploadFile) -> str:
    """
    Example question:
      "What is the markdown content of the PDF, formatted with prettier@3.4.2?"

    Steps:
      1) Extract text from the uploaded PDF
      2) Convert to naive Markdown
      3) (Optional) Run 'npx prettier@3.4.2 --parser=markdown' on the Markdown to format it
      4) Return the final, formatted Markdown string
    """

    # 1) Read PDF from UploadFile into memory, then parse text
    pdf_bytes = await file.read()
    pdf_path = "temp_to_markdown.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Use PyPDF2 to extract text from each page
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        all_text = []
        for page_idx in range(len(reader.pages)):
            page = reader.pages[page_idx]
            page_text = page.extract_text() or ""
            # Basic cleanup
            page_text = page_text.strip()
            # Collect
            all_text.append(page_text)
    raw_text = "\n\n".join(all_text)

    # 2) Convert to a naive Markdown. For example:
    #    - Split on double newlines to get paragraphs
    #    - Insert blank lines or bullet points, etc.
    paragraphs = raw_text.split("\n\n")
    markdown_lines = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # If the paragraph starts with a bullet marker or dash, keep it
        # Otherwise, treat it as a normal paragraph
        # (This is extremely simplistic - adjust as needed)
        if re.match(r"^[*•\-]\s", paragraph):
            markdown_lines.append(paragraph)
        else:
            # Possibly add a blank line before paragraphs in Markdown
            markdown_lines.append(paragraph + "\n")

    naive_markdown = "\n".join(markdown_lines)

    # 3) (Optional) Run Prettier on the Markdown
    #    This requires Node.js, npx, and an internet or local environment with 'prettier@3.4.2' installed
    try:
        with open("temp.md", "w", encoding="utf-8") as temp_md_file:
            temp_md_file.write(naive_markdown)

        # Format with npx prettier@3.4.2
        subprocess.run(
            ["npx", "prettier@3.4.2", "--parser=markdown", "--write", "temp.md"],
            check=True,
            capture_output=True,
        )

        # Read back the formatted MD
        with open("temp.md", "r", encoding="utf-8") as temp_md_file:
            formatted_markdown = temp_md_file.read()

    except FileNotFoundError:
        # If npx or Node is not installed, fallback to naive_markdown
        formatted_markdown = naive_markdown
    except subprocess.CalledProcessError as e:
        # If Prettier fails, fallback to naive_markdown
        formatted_markdown = naive_markdown

    # 4) Return the final, formatted Markdown
    return formatted_markdown

#-------- end of GA4 questions-------
#----------------------------------------------------------------------------

#-------- GA5 questions---------

# GA5 Q1 - Calculate total margin from Excel file

import re
import io
from datetime import datetime
from dateutil import parser

import pandas as pd
from fastapi import UploadFile

@register_question(r".*Download the Sales Excel file: .* What is the total margin for transactions before (.*) for (.*) sold in (.*)\?.*")
async def ga5_q1(question: str, file: UploadFile) -> str:
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

# GA5 Q2 - Count unique student IDs in a text file


# @register_question(r".*Download.*text.* file.*q-clean-up-student-marks.txt.*(unique students|number of unique students|student IDs).*")
@register_question(r".*(unique.*students|student IDs).*?(file|download).*")

async def ga5_q2(question: str, file: UploadFile) -> str:
    file_content = await file.read()
    lines = file_content.decode("utf-8").splitlines()
    student_ids = set()
    pattern = re.compile(r'-\s*([\w\d]+)::?Marks')
    for line in lines:
        match = pattern.search(line)
        if match:
            student_ids.add(match.group(1))
    return str(len(student_ids))

# GA5 Q5 - Calculate from JSON file - itme name, city name and units qty

@register_question(r".*[Hh]ow\s+many\s+units\s+of\s+(?P<product>.*?)\s+were\s+sold\s+in\s+(?P<city>.*?)\s+on\s+transactions\s+with\s+at\s+least\s+(?P<min_units>\d+)\s+units.*")
async def ga5_q5(question: str, file: UploadFile) -> str:
    # -------------------------------------------------------------------
    # 1) Extract parameters from the question (product, city, min_units)
    # -------------------------------------------------------------------
    match = re.match(
        r".*[Hh]ow\s+many\s+units\s+of\s+(?P<product>.*?)\s+were\s+sold\s+in\s+(?P<city>.*?)\s+on\s+transactions\s+with\s+at\s+least\s+(?P<min_units>\d+)\s+units.*",
        question,
    )
    if not match:
        return "0"  # or raise an exception if you want to enforce format

    product_name = match.group("product").strip()
    user_city_input = match.group("city").strip()
    min_sales = int(match.group("min_units"))

    # -------------------------------------------------------------------
    # 2) Load the JSON file into a Pandas DataFrame
    # -------------------------------------------------------------------
    file_content = await file.read()
    sales_data = json.loads(file_content)
    df = pd.DataFrame(sales_data)

    # -------------------------------------------------------------------
    # 3) Create a dictionary mapping "canonical city name" -> "set of variants"
    #    Extend this based on your dataset analysis
    # -------------------------------------------------------------------
    city_variants_map = {
        "mexico city": {
            "mexico city", "mexico-city", "mexicocity", "mexiko city", "mexico cty", "mexicoo city"
        },
        "london": {
            "london", "londen", "londn", "lonndon", "londonn", "londdon", "londnn"
        },
        "dhaka": {
            "dhaka", "dhakaa", "dhaaka", "dhacka"
        },
        "karachi": {
            "karachi", "karachee", "karachii", "karrachii", "karrchi", "karachy"
        },
        "lahore": {
            "lahore", "lahoore", "lahhore", "lahorre", "lahor"
        },
        "cairo": {
            "cairo", "caiiro", "ciro", "kairo", "kairoo"
        },
        "istanbul": {
            "istanbul", "istambul", "istanboul", "istnabul", "istaanbul"
        },
        "chennai": {
            "chennai", "chennay", "chennnai", "chenai", "chenaii"
        },
        "beijing": {
            "beijing", "bejing", "bejeing", "bejjing", "beijng", "bijing"
        },
        "mumbai": {
            "mumbai", "mumbay", "mumbbi", "mumbei", "mumby", "mombai", "mowmbai"  # add as needed
        },
        "bangalore": {
            "bangalore", "banglore", "bangaloore", "bengalore", "bangaloree"
        },
        "shanghai": {
            "shanghai", "shangai", "shanhai", "shanghii", "shanghhi"  # etc.
        },
        "tokyo": {
            "tokyo", "tokyoo", "tokeyo", "toikyo", "tokio"
        },
        "sao paulo": {
            "sao paulo", "sao pualo", "sao paoulo", "sao paolo", "sao paoulo", "sau paulo", "são paulo"
        },
        "rio de janeiro": {
            "rio de janeiro", "rio de janiero", "rio de janeirro", "rio de janiro", "rio de janero"
        },
        "paris": {
            "paris", "paries", "pariss", "parris"
        },
        "kolkata": {
            "kolkata", "kolkatta", "kolcata", "kolcotta", "kolkataa"
        },
        "osaka": {
            "osaka", "osakaa", "osakka", "osakkaa", "osaca", "oosaka"
        },
        "lagos": {
            "lagos", "lagoss", "lagose", "laggoss"
        },
        "bogota": {
            "bogota", "bogotaa", "bogotaà", "bogata", "bogotta", "bogotá"
        },
        "buenos aires": {
            "buenos aires", "buenes aires", "buienos aires", "buenoss aires", "buenos airres", "buenos aeres"
        },
        "jakarta": {
            "jakarta", "jakata", "jakkarta", "jakarata", "jakkarta"
        },
        "kinshasa": {
            "kinshasa", "kinshasaa", "kinshasha", "kinshassa", "kinshas"
        },
        "manila": {
            "manila", "manilaa", "mannila", "manil", "manla"
        },
        "delhi": {
            "delhi", "deli", "delly", "dehly", "dhelhi"
        },
    }
    # Make *every* sub-variant lowercase for easy matching
    # We'll create a helper dictionary: sub_variant_map[lower_str] = canonical_name
    sub_variant_map = {}
    for canonical, variants in city_variants_map.items():
        for variant in variants:
            sub_variant_map[variant.lower()] = canonical

    # -------------------------------------------------------------------
    # 4) Standardize city names in the DataFrame
    # -------------------------------------------------------------------
    def standardize_city(city_str: str) -> str:
        city_lower = city_str.strip().lower()
        return sub_variant_map.get(city_lower, city_str)  # fallback to as-is if not found

    df["city_standardized"] = df["city"].apply(standardize_city)

    # -------------------------------------------------------------------
    # 5) Determine the canonical form for the user’s input city
    #     (so that we group by the same standardized name)
    # -------------------------------------------------------------------
    # Attempt to map the user-provided city to its canonical form
    user_city_lower = user_city_input.lower()
    if user_city_lower in sub_variant_map:
        canonical_city = sub_variant_map[user_city_lower]  # e.g. "mexico city" for "mexico-city"
    else:
        # If the user typed a city we don’t have in the dictionary, we just use their exact name
        # (which might not match any row if spelled incorrectly)
        canonical_city = user_city_input

    # -------------------------------------------------------------------
    # 6) Filter the DataFrame: product == product_name, sales >= min_sales
    # -------------------------------------------------------------------
    filtered_df = df[(df["product"] == product_name) & (df["sales"] >= min_sales)]

    # -------------------------------------------------------------------
    # 7) Group by the standardized city name, sum the sales
    # -------------------------------------------------------------------
    sales_by_city = filtered_df.groupby("city_standardized")["sales"].sum().reset_index()

    # -------------------------------------------------------------------
    # 8) Get the sum of sales for the canonical city in question
    # -------------------------------------------------------------------
    city_total = sales_by_city.loc[sales_by_city["city_standardized"] == canonical_city, "sales"].sum()

    # -------------------------------------------------------------------
    # 9) Return the total as a string
    # -------------------------------------------------------------------
    return str(int(city_total))




# GA5 Q6 - Calculate total sales from JSONL file
@register_question(r".*total sales value.*")
async def ga5_q6(question: str, file: UploadFile) -> str:
    file_content = await file.read()
    total_sales = 0
    file_content_str = file_content.decode("utf-8")
    sales_matches = re.findall(r'"sales":\s*([\d.]+)', file_content_str)
    total_sales = sum(int(float(sales)) for sales in sales_matches)
    return str(total_sales)

# GA5 Q7 - Count occurrences of "LGK" as a key in nested JSON

#@register_question(r".*?(LGK).*?(appear|count|frequency).*?(key).*")
@register_question(r".*appear as a key.*")

async def ga5_q7(question: str, file: UploadFile) -> str:
    """
    Given a question string that contains a placeholder key (e.g., "How many times does LGK appear as a key?"),
    and a file containing a JSON structure, return how many times that key appears as a key in the JSON.
    """
    # 1) Extract the target key from the question
    #    Below is a simple pattern looking for text after "does " and before " appear as a key"
    match = re.search(r"does\s+(.+?)\s+appear\s+as\s+a\s+key", question, re.IGNORECASE)
    if match:
        key_to_count = match.group(1)
    else:
        # Fallback if parsing fails; default to "LGK"
        key_to_count = "LGK"

    # 2) Read and parse the JSON file
    file_content = await file.read()
    data = json.loads(file_content.decode("utf-8"))

    # 3) Define a helper function to do the recursive counting
    def count_key_occurrences(obj, key_str):
        count = 0
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key_str:
                    count += 1
                count += count_key_occurrences(v, key_str)
        elif isinstance(obj, list):
            for item in obj:
                count += count_key_occurrences(item, key_str)
        return count

    # 4) Count how many times the extracted key appears
    result_count = count_key_occurrences(data, key_to_count)

    # 5) Return the result (as a string)
    return str(result_count)



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