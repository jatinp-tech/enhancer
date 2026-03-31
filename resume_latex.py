import sys
from pathlib import Path
from typing import Optional

try:
    from google import genai
except ImportError:
    print("Error: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

API_KEY = "AIzaSyBrdrpa63BZzh73ZKV-S10D4NWZHZfYD2g"
GEMINI_MODEL = "gemini-3-flash-preview"
MARKDOWN_ARTIFACTS = ["```latex", "```", "```python", "```text"]

PROMPT_TEMPLATE = """You are a LaTeX resume optimizer. Output ONLY raw LaTeX code. NO MARKDOWN.

==== FORBIDDEN (VIOLATION = INVALID OUTPUT) ====
- NEVER output ** ** or __ __ anywhere
- NEVER output ``` or ### or markdown of any kind
- NEVER output bullets outside \\item or LaTeX lists
- If your output contains **, __, or ```, it FAILS

==== REQUIRED OUTPUT ====
Return the COMPLETE LaTeX resume with these modifications:

FOR TEXT EMPHASIS:
- Use \\textbf{{text}} NOT **text**
- Use \\textit{{text}} NOT __text__  
- Example: \\textbf{{Python}} and \\textit{{important}}

FOR LISTS/BULLETS:
- Use existing \\resumeItemListStart and \\item commands
- Do NOT create new bullets with - or *
- Keep exact LaTeX structure

==== CORE WORKFLOW ====
- Extract relevant keywords from JD (ignore metadata/headers)
- Identify overlapping skills between JD and candidate's resume
- Enhance summary and skills with JD-relevant keywords
- Rewrite experience bullets to match JD language naturally
- Keep ALL LaTeX structure, commands, and definitions intact

FOR OPTIMIZATION:
1. Analyze JD keywords against candidate's actual skills:
   - High Expertise: Python, Machine Learning (ML)
   - Good Expertise: Computer Vision
   - Medium Expertise: SQL
   - Low/Knowledge Only: PySpark, GenAI (has knowledge, but no direct experience)
   - STRICT RULE: Do completely NOT add keywords or skills that are not explicitly present in the provided Job Description text.

2. Enhance Summary:
   Replace generic text with JD-specific keywords and years.
   
3. Enhance Skills Section:
   Add missing skills from JD ONLY if they align with the candidate's actual skills listed above.
   Organize: Programming, Tools, ML/CV, Domain Skills.
   
4. Enhance Experience:
   Use JD keywords in existing bullet points.
   CRITICAL: Modify ONLY plain text keywords to prevent breaking PDF. Do NOT add or remove \\item commands.
   
5. Keep All Definitions & Structure:
   ALL \\newcommand definitions must be in output unchanged.
   Do NOT remove or modify any LaTeX command definitions.

==== ABSOLUTELY CRITICAL ====
Your entire output is INVALID if it contains:
✗ ** or **text** (markdown bold)
✗ __ or __text__ (markdown italic)  
✗ ``` (code blocks)
✗ # ## ### (headers)
✗ - or * bullets outside \\item

JOB DESCRIPTION:
{jd}

RESUME LATEX (with ALL \\newcommand definitions):
{resume}

OUTPUT ONLY the modified LaTeX code:"""


def load_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise IOError(f"Error reading {path}: {e}")


def save_file(path: str, content: str) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        print(f"✅ Saved: {path}")
    except Exception as e:
        raise IOError(f"Error writing to {path}: {e}")


def clean_markdown(text: str) -> str:
    for artifact in MARKDOWN_ARTIFACTS:
        text = text.replace(artifact, "")
    return text.strip()


def build_prompt(jd: str, resume: str) -> str:
    return PROMPT_TEMPLATE.format(jd=jd, resume=resume)


def optimize_resume(jd: str, resume: str, api_key: Optional[str] = None) -> str:
    if not api_key:
        api_key = API_KEY
    
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(jd, resume)
    print(f"🔄 Calling {GEMINI_MODEL}...")
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    optimized = clean_markdown(response.text)
    print(f"✅ Optimization complete")
    return optimized


def main(jd_path: str, resume_path: str, output_path: str, api_key: Optional[str] = None) -> None:
    try:
        print(f"📄 Loading JD: {jd_path}")
        jd = load_file(jd_path)
        print(f"📄 Loading resume: {resume_path}")
        resume = load_file(resume_path)
        optimized = optimize_resume(jd, resume, api_key)
        save_file(output_path, optimized)
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimize LaTeX resume for job description using Gemini AI")
    parser.add_argument("--jd", required=True, help="Path to job description file")
    parser.add_argument("--resume", required=True, help="Path to LaTeX resume file")
    parser.add_argument("--output", required=True, help="Path to save optimized resume")
    parser.add_argument("--api-key", help="Gemini API key (default: GEMINI_API_KEY env var)")
    args = parser.parse_args()
    main(jd_path=args.jd, resume_path=args.resume, output_path=args.output, api_key=args.api_key)

