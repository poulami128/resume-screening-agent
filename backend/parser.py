# backend/parser.py
import re
from typing import List, Dict

# simple skills list — customize with your domain terms
SKILLS_LIST = [
    "python", "django", "flask", "pandas", "numpy", "sql", "aws", "docker", "kubernetes",
    "machine learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "rest", "api"
]

def parse_job_description(jd_text: str) -> Dict:
    if not jd_text:
        return {"skills": [], "years": None}
    text = jd_text.lower()
    skills = extract_skills_from_text(text)
    # naive years extraction
    m = re.search(r'(\d+)[\+]? years?', text)
    years = int(m.group(1)) if m else None
    return {"skills": skills, "years": years}

def extract_skills_from_text(text: str) -> List[str]:
    found = []
    for s in SKILLS_LIST:
        if s in text:
            found.append(s)
    return found
