import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import spacy
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from fastapi import Body

load_dotenv()
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
configure(api_key=os.getenv("GOOGLE_API_KEY"))

nlp = spacy.load("en_core_web_sm")
# Use only genai.generate_content for Gemini API calls

import json

def gemini_resume_job_matching(resume_text: str, job_description: str, google_job_description: str):
    """
    Uses Gemini AI to extract skills and compute match percentages between a resume,
    a job description, and a Google job description.
    Returns a dict with skills and match details for both jobs.
    """
    prompt = f"""
Given the following three texts:

Resume:
{resume_text}

Job Description:
{job_description}

Google Job Description:
{google_job_description}

1. Extract the relevant skills from each.
2. Compute a match percentage between the resume and the Job Description.
3. Compute a match percentage between the resume and the Google Job Description.
4. List the matched and missing skills for each comparison.

Respond in the following JSON format:
{{
  "resumeSkills": [...],
  "jobDescriptionSkills": [...],
  "googleJobDescriptionSkills": [...],
  "matchWithJobDescription": {{
    "matchedSkills": [...],
    "missingSkills": [...],
    "matchPercentage": <number between 0 and 100>
  }},
  "matchWithGoogleJobDescription": {{
    "matchedSkills": [...],
    "missingSkills": [...],
    "matchPercentage": <number between 0 and 100>
  }}
}}
"""
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(prompt)
    # Extract JSON from the response
    try:
        start = response.text.find('{')
        end = response.text.rfind('}') + 1
        json_str = response.text[start:end]
        result = json.loads(json_str)
        return result
    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Full response:", response.text)
        return None

app = FastAPI()

with open(os.path.join(os.path.dirname(__file__), "skills_list.txt")) as f:
    KNOWN_SKILLS = set(line.strip().lower() for line in f if line.strip())

class JobDescription(BaseModel):
    id: Optional[str] = None
    googleJobId: Optional[str] = None  # changed from jobId
    title: Optional[str] = None
    companyName: Optional[str] = None
    location: Optional[str] = None
    via: Optional[str] = None
    shareLink: Optional[str] = None
    postedAt: Optional[str] = None
    salary: Optional[str] = None
    scheduleType: Optional[str] = None
    qualifications: Optional[str] = None
    description: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    benefits: Optional[List[str]] = None
    applyLinks: Optional[str] = None
    createdDateTime: Optional[str] = None
    lastUpdatedDateTime: Optional[str] = None
    # jobTitle and city are objects, not used for skill extraction

class ResumeAnalysisRequest(BaseModel):
    resume_text: str
    jobs: List[JobDescription]

class JobMatchResult(BaseModel):
    job_id: str
    job_title: str
    company: str
    match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    missing_certifications: List[str]
    missing_education: Optional[str]
    ai_suggestions: str
    location: Optional[str] = None
    salary: Optional[str] = None
    scheduleType: Optional[str] = None
    qualifications: Optional[str] = None
    description: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    benefits: Optional[List[str]] = None
    applyLink: Optional[str] = None

class ResumeAnalysisResponse(BaseModel):
    top_matches: List[JobMatchResult]

class ResumeImproveRequest(BaseModel):
    resume_text: str
    job: JobDescription
    suggestion: str

class ResumeImproveResponse(BaseModel):
    improved_resume: str

def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found = set()
    for skill in KNOWN_SKILLS:
        if skill in text_lower:
            found.add(skill)
    print("Extracted skills from text:", found)  # <-- Add this line
    return found

def extract_job_skills(job: dict) -> set:
    # Combine qualifications, description, and responsibilities for skill extraction
    text = (job.get("qualifications") or "") + " " + (job.get("description") or "")
    responsibilities = job.get("responsibilities")
    if responsibilities and isinstance(responsibilities, list):
        text += " " + " ".join(responsibilities)
    return extract_skills(text)

def call_gemini_suggestions(job, missing_skills, missing_qualifications, missing_experience):
    prompt = (
        f"A candidate is applying for the job '{job.get('title', '')}' at '{job.get('companyName', '')}'.\n"
        f"Their resume is missing these skills: {', '.join(missing_skills) or 'None'}, "
        f"qualifications: {missing_qualifications or 'None'}, "
        f"and experience: {missing_experience or 'None'}.\n"
        "Suggest what they should add or improve in their resume to reach a 100% match for this job."
    )
    model = GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    print("Prompt sent to Gemini:", prompt)
    print("Gemini suggestion:", response.text if hasattr(response, 'text') else response)
    return response.text.strip() if hasattr(response, 'text') and response.text else str(response)

@app.post("/analyze", response_model=ResumeAnalysisResponse)
def analyze_resume(request: ResumeAnalysisRequest):
    resume_skills = extract_skills(request.resume_text)
    print("Resume skills:", resume_skills)
    results = []

    for job in request.jobs:
        job_dict = job.dict() if hasattr(job, 'dict') else dict(job)
        # Get Google Job Description if present, else use empty string
        google_job_desc = job_dict.get('googleJobDescription', '') or ''
        # Use Gemini to get AI-based skills and match percentage
        gemini_result = gemini_resume_job_matching(
            request.resume_text,
            job_dict.get('description', '') or '',
            google_job_desc
        )
        if gemini_result:
            matched_skills = gemini_result["matchWithJobDescription"]["matchedSkills"]
            missing_skills = gemini_result["matchWithJobDescription"]["missingSkills"]
            match_percentage = gemini_result["matchWithJobDescription"]["matchPercentage"]
        else:
            # fallback to old logic if Gemini fails
            resume_skills = extract_skills(request.resume_text)
            job_skills = extract_job_skills(job_dict)
            matched_skills = list(resume_skills & job_skills)
            missing_skills = list(job_skills - resume_skills)
            match_percentage = 100 * len(matched_skills) / max(1, len(job_skills)) if job_skills else 0.0

        missing_qualifications = job_dict.get('qualifications', None)
        missing_experience = job_dict.get('description', None)  # You can improve this logic

        ai_suggestions = None
        if match_percentage < 100:
            ai_suggestions = call_gemini_suggestions(
                job_dict, missing_skills, missing_qualifications, missing_experience
            )

        results.append(JobMatchResult(
            job_id=job_dict.get('googleJobId') or job_dict.get('id', ''),
            job_title=job_dict.get('title', ''),
            company=job_dict.get('companyName', ''),
            match_percentage=round(match_percentage, 2),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            missing_certifications=[],  # Add if you extract these
            missing_education=None,     # Add if you extract these
            ai_suggestions=ai_suggestions or "",
            location=job_dict.get('location'),
            salary=job_dict.get('salary'),
            scheduleType=job_dict.get('scheduleType'),
            qualifications=job_dict.get('qualifications'),
            description=job_dict.get('description'),
            responsibilities=job_dict.get('responsibilities'),
            benefits=job_dict.get('benefits'),
            applyLink=job_dict.get('applyLinks') or job_dict.get('applyLink'),
        ))

    results.sort(key=lambda x: x.match_percentage, reverse=True)
    return ResumeAnalysisResponse(top_matches=results[:5])

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
from docx import Document

@app.post("/improve")
async def improve_resume_proxy(request: Request):
    data = await request.json()
    resume_text = data.get("resumeText", "")
    suggestion = data.get("suggestion", "")
    # Optionally accept job info if available
    job_id = data.get("googleJobId", "")
    job_seeker_id = data.get("jobSeekerId", "")

    prompt = (
        f"Here is a resume:\n{resume_text}\n\n"
        f"Please update the resume to address this suggestion: '{suggestion}'. "
        "Make the resume more relevant for the job, but keep it realistic and concise. "
        "Return only the improved resume text."
    )
    model = GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    improved_resume = response.text.strip() if hasattr(response, 'text') and response.text else str(response)

    # Compute match percentage using Gemini
    # For this, we need a job description. We'll use the suggestion as a proxy if job description is not provided.
    job_description = suggestion  # You can adjust this if you have a real job description field
    gemini_result = gemini_resume_job_matching(improved_resume, job_description, "")
    if gemini_result:
        match_percentage = gemini_result["matchWithJobDescription"]["matchPercentage"]
    else:
        match_percentage = None

    # Get AI suggestions for further improvement
    ai_suggestion = call_gemini_suggestions({"title":"","companyName":""}, [], suggestion, None)

    return JSONResponse({
        "resumeText": improved_resume,
        "matchPercentage": match_percentage,
        "suggestions": [ai_suggestion] if ai_suggestion else [],
        "canDownload": False
    })

# --- NEW ENDPOINT: Download resume as Word document ---
# Frontend should POST {"resume_text": "..."} to /download_resume when progress is 100%,
# then offer the returned file as a download link to the user.
@app.post("/download_resume")
async def download_resume(request: Request):
    data = await request.json()
    resume_text = data.get("resume_text", "")
    doc = Document()
    for line in resume_text.splitlines():
        doc.add_paragraph(line)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    headers = {
        'Content-Disposition': 'attachment; filename="updated_resume.docx"'
    }
    return StreamingResponse(buffer, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', headers=headers)