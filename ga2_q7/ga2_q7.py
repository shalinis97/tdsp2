from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api")
async def get_marks(
    name: List[str] = Query(..., description="Names to fetch marks for"),
    file: UploadFile = File(...)
):
    """
    Accepts a GET request with:
    - Query parameters: name=X&name=Y
    - UploadFile: JSON file with student marks

    Returns:
    - {"marks": [marks_of_X, marks_of_Y]}
    """
    # Read and parse uploaded JSON file
    try:
        contents = await file.read()
        data = json.loads(contents)
        marks_dict = {entry["name"]: entry["marks"] for entry in data}
    except Exception as e:
        return {"error": f"Failed to read or parse file: {str(e)}"}

    # Collect marks for each requested name
    results = [marks_dict.get(n, None) for n in name]
    return {"marks": results}
