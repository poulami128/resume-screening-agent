# ingest_resumes.py
import os
from backend.parser import extract_resume_text

def load_resumes_from_folder(folder="data/resumes"):
    resumes = []
    if not os.path.exists(folder):
        print("No resumes folder found. Create data/resumes/ and add files.")
        return resumes
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isdir(path):
            continue
        try:
            text = extract_resume_text(path)
            resumes.append({"path": path, "text": text})
        except Exception as e:
            print("Failed to parse:", path, e)
    return resumes

if __name__ == "__main__":
    rs = load_resumes_from_folder()
    print(f"Loaded {len(rs)} resumes")
