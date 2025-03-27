from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_wikipedia_url(country: str) -> str:
    return f"https://en.wikipedia.org/wiki/{country.replace(' ', '_')}"

def extract_headings_from_html(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    headings = []
    for level in range(1, 7):
        for tag in soup.find_all(f'h{level}'):
            text = tag.get_text(strip=True)
            if text:
                headings.append((level, text))
    return headings

def generate_markdown_outline(country: str, headings: list) -> str:
    md = "## Contents\n\n"
    md += f"# {country}\n\n"
    for level, heading in headings:
        md += f"{'#' * level} {heading}\n\n"
    return md

@app.get("/api/outline", response_class=PlainTextResponse)
async def get_country_outline(country: str = Query(..., description="Country name")):
    if not country:
        raise HTTPException(status_code=400, detail="Country parameter is required")

    url = get_wikipedia_url(country)

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Could not fetch page for '{country}': {e}")

    headings = extract_headings_from_html(response.text)
    if not headings:
        raise HTTPException(status_code=404, detail=f"No headings found for {country}")

    markdown_outline = generate_markdown_outline(country, headings)
    return markdown_outline

# Optional if you're running as script
if __name__ == "__main__":
    uvicorn.run("ga4_q3:app", host="0.0.0.0", port=8000, reload=True)



#uvicorn ga4_q3:app --host 127.0.0.1 --port 8003 --reload                                           │                                                                     │
                            