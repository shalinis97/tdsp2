from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from io import StringIO
import pandas as pd
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api")
async def upload_csv(
    file: UploadFile = File(...),
    class_name: Optional[List[str]] = Query(None, alias="class")
):
    """
    Accept a CSV file and return student data.
    Optional class filters via query string: /api?class=1A&class=1B
    """
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    if class_name:
        df = df[df["class"].isin(class_name)]

    students = df.to_dict(orient="records")
    return {"students": students}

# Run the app directly with Uvicorn if this file is executed
if __name__ == "__main__":
    uvicorn.run("ga2_q9:app", host="127.0.0.1", port=8007, reload=True)
