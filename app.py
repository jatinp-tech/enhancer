import streamlit as st
from google import genai
import sys

# Pull the API key securely from Streamlit Secrets, with a fallback for local testing
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    API_KEY = "" # Or leave this blank

MARKDOWN_ARTIFACTS = ["```latex", "```", "```python", "```text"]

PROMPT_TEMPLATE = """You are a LaTeX resume optimizer. Output ONLY raw LaTeX code. NO MARKDOWN.

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

def clean_markdown(text: str) -> str:
    for artifact in MARKDOWN_ARTIFACTS:
        text = text.replace(artifact, "")
    return text.strip()

def optimize_resume(jd: str, resume: str, model_id: str) -> str:
    client = genai.Client(api_key=API_KEY)
    prompt = PROMPT_TEMPLATE.format(jd=jd, resume=resume)
    response = client.models.generate_content(model=model_id, contents=prompt)
    return clean_markdown(response.text)

# --- STREAMLIT UI ---
st.set_page_config(page_title="LaTeX Resume Optimizer", page_icon="📄")
st.title("📄 LaTeX Resume Optimizer")

st.markdown("ATS resume optimizer with JD matching using Google AI Studio selected models.")

MODEL_OPTIONS = {
    "gemini-3.0-flash": "1. Gemini 3 Flash (Max 20 Requests/Day | The Best, won't break LaTeX)",
    "gemini-2.5-flash": "2. Gemini 2.5 Flash (Max 20 Requests/Day | Highly Capable alternative)",
    "gemini-3.1-flash-lite": "3. Gemini 3.1 Flash Lite (Max 500 Requests/Day | Best for bulk testing)"
}

selected_model = st.selectbox(
    "Choose your AI Model:",
    options=list(MODEL_OPTIONS.keys()),
    format_func=lambda x: MODEL_OPTIONS[x],
    index=0
)

jd_input = st.text_area("Paste Job Description (JD) here:", height=300)

if st.button("Generate Optimized Resume", type="primary"):
    if not jd_input:
        st.warning("⚠️ Please provide a Job Description.")
    else:
        with st.spinner("Optimizing your resume with Gemini..."):
            try:
                with open("resume.tex", "r", encoding="utf-8") as f:
                    resume_content = f.read()
            except FileNotFoundError:
                st.error("❌ 'resume.tex' was not found in the repository! Make sure it is pushed to GitHub.")
                st.stop()
            try:
                optimized_tex = optimize_resume(jd_input, resume_content, selected_model)
                st.success(f"✅ Optimization complete using {selected_model}!")
                
                with st.spinner("Compiling LaTeX to PDF..."):
                    import tempfile
                    import subprocess
                    import os
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        tex_path = os.path.join(temp_dir, "optimized.tex")
                        pdf_path = os.path.join(temp_dir, "optimized.pdf")
                        
                        with open(tex_path, "w", encoding="utf-8") as f:
                            f.write(optimized_tex)
                        
                        compile_process = subprocess.run(
                            ["pdflatex", "-interaction=nonstopmode", "optimized.tex"],
                            cwd=temp_dir,
                            capture_output=True,
                            text=True
                        )
                        
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                pdf_data = f.read()
                                
                            st.success("🎉 PDF Compiled Successfully!")
                            st.download_button(
                                label="⬇️ Download Optimized PDF",
                                data=pdf_data,
                                file_name="optimized.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("❌ Failed to compile LaTeX to PDF. The model likely generated invalid LaTeX structure.")
                            with st.expander("View LaTeX Errors"):
                                st.text(compile_process.stdout)
                
                st.download_button(
                    label="⬇️ Download raw optimized.tex",
                    data=optimized_tex,
                    file_name="optimized.tex",
                    mime="text/plain"
                )
                
                with st.expander("Preview Optimized LaTeX source"):
                    st.code(optimized_tex, language="latex")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error("🛑 Rate Limit Exceeded for this Model!")
                    st.warning("💡 **Tip:** You have hit the daily free quota for this specific AI model. Please scroll up and select a different model (e.g. Gemini 3.1 Flash Lite or Gemma) from the dropdown menu to continue!")
                else:
                    st.error(f"An error occurred: {error_msg}")


