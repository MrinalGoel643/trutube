"""
TruTube Backend - YouTube Video Fact-Checker API
AIPI Mini Hackathon #2: Can Machines Understand Us Reliably?

Stack: FastAPI + OpenAI (GPT-4o-mini) + DuckDuckGo Search (free, no key needed)
"""

import json
import re
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from duckduckgo_search import DDGS

# == App Setup =================================================================

app = FastAPI(title="TruTube API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
MODEL = "gpt-4o-mini"

# == JSON File Cache ===========================================================
# Persists across server restarts. Pre-warm demo videos with POST /warm-up.
# Cache lives in cache/ directory as individual JSON files per video.

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
STRESS_CACHE_FILE = CACHE_DIR / "_stress_test.json"


def _get_cached_analysis(video_id: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{video_id}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def _save_analysis_cache(video_id: str, data: dict):
    cache_file = CACHE_DIR / f"{video_id}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def _get_cached_stress_test() -> Optional[dict]:
    if STRESS_CACHE_FILE.exists():
        with open(STRESS_CACHE_FILE, "r") as f:
            return json.load(f)
    return None


def _save_stress_test_cache(data: dict):
    with open(STRESS_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


# == Schemas ===================================================================

class AnalyzeRequest(BaseModel):
    youtube_url: str

class WarmUpRequest(BaseModel):
    youtube_urls: list[str]

class VerifiedClaim(BaseModel):
    claim: str
    category: str
    has_hedging: bool
    verdict: str
    confidence: float
    evidence_summary: str
    sources: list[str]

class AnalyzeResponse(BaseModel):
    video_id: str
    word_count: int
    claims: list[VerifiedClaim]
    factuality_score: float
    summary: dict
    cached: bool = False

class StressTestCase(BaseModel):
    input_text: str
    test_type: str
    should_extract: bool
    description: str
    extracted: bool
    correct: bool
    claims_found: list[dict]

class StressTestResponse(BaseModel):
    results: list[StressTestCase]
    precision: float
    cached: bool = False


# == Core Logic ================================================================

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|&v=)([^#&?\s]{11})",
        r"^([^#&?\s]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


def get_transcript(video_id: str) -> str:
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    return " ".join(seg.text for seg in transcript)


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    response = llm.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def parse_json_response(text: str):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def extract_claims(transcript: str) -> list[dict]:
    system = "You are a fact-checking assistant. You only output valid JSON."
    user = (
        "Extract verifiable factual claims from this transcript.\n\n"
        "RULES:\n"
        "- Only extract claims checkable against real-world facts "
        "(statistics, dates, scientific facts, named events, etc.)\n"
        "- IGNORE opinions, subjective statements, jokes, sarcasm, "
        "rhetorical questions, and vague statements\n"
        "- Extract 5-15 of the most important/checkable claims\n"
        '- For each claim, note if it contains hedging language '
        '("some say", "I think", "probably")\n\n'
        "Return ONLY a JSON array. Each element:\n"
        '{"claim": "the factual claim as stated", '
        '"category": "science|history|statistics|health|politics|economics|other", '
        '"has_hedging": true/false}\n\n'
        "TRANSCRIPT:\n" + transcript[:8000]
    )
    text = call_llm(system, user)
    return parse_json_response(text)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
            for r in results
        ]
    except Exception:
        return []


def verify_claim(claim: str) -> dict:
    search_results = search_web(claim, max_results=5)
    evidence_text = "\n\n".join(
        "[" + r["title"] + "]\n" + r["body"] + "\nURL: " + r["href"]
        for r in search_results
    )
    if not evidence_text.strip():
        evidence_text = "(No search results found)"

    system = "You are a rigorous fact-checker. You only output valid JSON."
    user = (
        'CLAIM: "' + claim + '"\n\n'
        "SEARCH EVIDENCE:\n" + evidence_text + "\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"verdict": "supported" or "refuted" or "unverifiable", '
        '"confidence": <float 0.0-1.0>, '
        '"evidence_summary": "<2-3 sentence summary>", '
        '"sources": ["source 1", "source 2"]}\n\n'
        "Rules:\n"
        '- "supported" = evidence clearly confirms the claim\n'
        '- "refuted" = evidence clearly contradicts (even if partially true)\n'
        '- "unverifiable" = not enough evidence to decide\n'
        "- Partially true claims with material inaccuracies = refuted\n"
        "- Sources = names/titles of pages that informed your verdict"
    )
    text = call_llm(system, user, max_tokens=1024)
    try:
        return parse_json_response(text)
    except Exception:
        return {
            "verdict": "unverifiable",
            "confidence": 0.0,
            "evidence_summary": "Non-parseable response.",
            "sources": [],
        }


def compute_factuality_score(results: list[dict]) -> float:
    if not results:
        return 0.0
    weights = {"supported": 1.0, "refuted": 0.0, "unverifiable": 0.4}
    total = sum(
        weights.get(r["verdict"], 0.4) * r.get("confidence", 0.5)
        for r in results
    )
    return round((total / len(results)) * 100, 1)


def _run_analysis(video_id: str) -> AnalyzeResponse:
    """Core analysis logic used by /analyze and /warm-up."""
    transcript = get_transcript(video_id)
    word_count = len(transcript.split())
    raw_claims = extract_claims(transcript)

    verified = []
    for c in raw_claims:
        result = verify_claim(c["claim"])
        verified.append(VerifiedClaim(
            claim=c["claim"],
            category=c.get("category", "other"),
            has_hedging=c.get("has_hedging", False),
            verdict=result.get("verdict", "unverifiable"),
            confidence=result.get("confidence", 0.0),
            evidence_summary=result.get("evidence_summary", ""),
            sources=result.get("sources", []),
        ))

    score = compute_factuality_score([v.model_dump() for v in verified])
    summary = {
        "supported": sum(1 for v in verified if v.verdict == "supported"),
        "refuted": sum(1 for v in verified if v.verdict == "refuted"),
        "unverifiable": sum(1 for v in verified if v.verdict == "unverifiable"),
        "total": len(verified),
    }
    return AnalyzeResponse(
        video_id=video_id,
        word_count=word_count,
        claims=verified,
        factuality_score=score,
        summary=summary,
    )


# == Stress Test Data ==========================================================

STRESS_TESTS = [
    {
        "input": "I mean, obviously the moon is made of cheese, right? *laughs*",
        "type": "Sarcasm",
        "should_extract": False,
        "description": "Sarcastic statement - should NOT be extracted",
    },
    {
        "input": "The Earth orbits the Sun at roughly 67,000 miles per hour.",
        "type": "Verifiable Fact",
        "should_extract": True,
        "description": "Clear factual claim - SHOULD be extracted",
    },
    {
        "input": "I personally think chocolate ice cream is the best flavor.",
        "type": "Opinion",
        "should_extract": False,
        "description": "Subjective opinion - should NOT be extracted",
    },
    {
        "input": "People say drinking 8 glasses of water a day is necessary, but who really knows.",
        "type": "Hedged Claim",
        "should_extract": True,
        "description": "Verifiable health claim with hedging - SHOULD be extracted",
    },
    {
        "input": "Einstein failed math in school and then became the greatest physicist ever.",
        "type": "Common Myth",
        "should_extract": True,
        "description": "Popular misconception - SHOULD be extracted and ideally refuted",
    },
    {
        "input": "You know what is wild? Like, the vibes in this economy are just... off.",
        "type": "Vague/Slang",
        "should_extract": False,
        "description": "Vague slang - should NOT be extracted",
    },
]


# == Endpoints =================================================================

@app.get("/health")
def health():
    cached_files = [f.stem for f in CACHE_DIR.glob("*.json") if f.stem != "_stress_test"]
    return {"status": "ok", "cached_videos": cached_files}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(req: AnalyzeRequest):
    """POST /analyze  |  Cached results return instantly."""
    video_id = extract_video_id(req.youtube_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Could not parse YouTube video ID")

    # Check JSON file cache
    cached_data = _get_cached_analysis(video_id)
    if cached_data:
        cached_data["cached"] = True
        return AnalyzeResponse(**cached_data)

    try:
        response = _run_analysis(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save to JSON file
    _save_analysis_cache(video_id, response.model_dump())
    return response


@app.post("/warm-up")
def warm_up(req: WarmUpRequest):
    """
    POST /warm-up
    Body: {"youtube_urls": ["url1", "url2", "url3"]}
    Pre-analyze videos before the demo. Run 10 min before your pitch!
    """
    results = {}
    for url in req.youtube_urls:
        video_id = extract_video_id(url)
        if not video_id:
            results[url] = {"status": "error", "detail": "Bad video ID"}
            continue
        if _get_cached_analysis(video_id):
            results[url] = {"status": "already_cached", "video_id": video_id}
            continue
        try:
            response = _run_analysis(video_id)
            _save_analysis_cache(video_id, response.model_dump())
            results[url] = {
                "status": "cached",
                "video_id": video_id,
                "factuality_score": response.factuality_score,
            }
        except Exception as e:
            results[url] = {"status": "error", "detail": str(e)}

    warmed = len([r for r in results.values() if r["status"] in ("cached", "already_cached")])
    return {"warmed_up": warmed, "results": results}


@app.get("/cache")
def view_cache():
    """See what is currently cached."""
    cached_videos = {}
    for f in CACHE_DIR.glob("*.json"):
        if f.stem == "_stress_test":
            continue
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            cached_videos[f.stem] = {
                "factuality_score": data.get("factuality_score"),
                "total_claims": data.get("summary", {}).get("total"),
            }
        except Exception:
            pass
    return {
        "cached_videos": cached_videos,
        "stress_test_cached": STRESS_CACHE_FILE.exists(),
    }


@app.delete("/cache")
def clear_cache():
    """Clear all cached results."""
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
    return {"status": "cache_cleared"}


@app.get("/transcript/{video_id}")
def get_video_transcript(video_id: str):
    """GET /transcript/{video_id}  |  Quick transcript preview."""
    try:
        transcript = get_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=422, detail="Could not fetch transcript: " + str(e))
    return {
        "video_id": video_id,
        "transcript": transcript,
        "word_count": len(transcript.split()),
    }


@app.post("/stress-test", response_model=StressTestResponse)
def run_stress_tests():
    """POST /stress-test  |  No body needed. Cached after first run."""
    cached = _get_cached_stress_test()
    if cached is not None:
        return StressTestResponse(**cached, cached=True)

    results = []
    for test in STRESS_TESTS:
        try:
            claims = extract_claims(test["input"])
            extracted = len(claims) > 0
            correct = extracted == test["should_extract"]
            results.append(StressTestCase(
                input_text=test["input"],
                test_type=test["type"],
                should_extract=test["should_extract"],
                description=test["description"],
                extracted=extracted,
                correct=correct,
                claims_found=claims,
            ))
        except Exception:
            results.append(StressTestCase(
                input_text=test["input"],
                test_type=test["type"],
                should_extract=test["should_extract"],
                description=test["description"],
                extracted=False,
                correct=not test["should_extract"],
                claims_found=[],
            ))

    precision = sum(1 for r in results if r.correct) / len(results) * 100
    cache_data = {
        "results": [r.model_dump() for r in results],
        "precision": precision,
    }
    _save_stress_test_cache(cache_data)
    return StressTestResponse(results=results, precision=precision)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)