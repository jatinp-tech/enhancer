import streamlit as st
from google import genai
import sys

# Fetch API key from Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not API_KEY:
    st.error("🔑 **API Key Missing!**")
    st.info("To fix this locally: Paste your key into `.streamlit/secrets.toml` like this:\n`GEMINI_API_KEY = 'your-key-here'`")
    st.stop()

MARKDOWN_ARTIFACTS = ["```latex", "```", "```python", "```text"]
PROMPT_TEMPLATE = """You are an elite LaTeX Resume Optimizer. Your mission is to adapt the candidate's resume for a specific Job Description (JD) with surgical precision. 

==== ABSOLUTE CONSTRANTS (CRITICAL) ====
1. ONE PAGE LIMIT: The output MUST stay on one page. Do NOT increase the word count. If you add text, you MUST delete original text to compensate.
2. NO BLOATING: Do NOT add new bullet points. SUMMARY must be max 3 lines.
3. RAW LATEX ONLY: Output ONLY pure LaTeX code. 
   - ✗ NO MARKDOWN BOLD (Do NOT use **keyword**)
   - ✗ NO MARKDOWN ITALIC (Do NOT use _keyword_)
   - ✗ NO CODE FENCES (Do NOT use ```latex or ```)
   - ⚠️ YOUR ENTIRE OUTPUT IS INVALID AND UNUSABLE IF IT CONTAINS ANY DOUBLE ASTERISKS (**).
4. PRESERVE STRUCTURE: Keep all LaTeX commands (\\newcommand, \\usepackage, \\geometry, \\vspace, \\item) EXACTLY intact. ONLY modify plain text.
5. STRICTLY FORBIDDEN: Do NOT include: Cloud Platforms (AWS, Azure, GCP), "Advanced Pipelines", or "Agentic AI". Explicitly ignore these.

==== CANDIDATE SKILL MATRIX ====
- CORE EXPERTISE (High): Python, Machine Learning (ML), ML Modeling, Data Analysis, POC R&D, Data Cleansing, Regression, feature engineering.
- SECONDARY (Good): Computer Vision (YOLOv8, Convolutional architectures).
- TOOLS/OTHER: SQL (Medium), PySpark, GenAI, LLMs, NLP, Basic RAG (Knowledge/POC level only).

==== OPTIMIZATION STRATEGY (ADD "GOOD POINTS") ====
1. SUMMARY: 
   - Rewrite into a 2-3 line technical hook. 
   - Format: "Machine Learning Engineer | 3 Years XP | Core Expertise in [JD Keyword 1], [JD Keyword 2], and Python."
   - Focus on "Model Reliability", "Performance Optimization", and "Validation" (key for AI QA roles).
2. EXPERIENCE BULLETS: 
   - Replace generic verbs with JD action verbs (e.g., "Validated", "Benchmarked", "Optimized", "Designed Frameworks").
   - MAP SKILLS: Use "Regression" and "feature engineering" to address JD needs like "Hallucination Detection" or "Model Quality Benchmarking".
   - Example Change: "Improved model reliability" -> "Developed regression analysis pipelines for model quality benchmarking and hallucination detection."
3. SKILLS SECTION:
   - Organize: Programming, ML Systems, Machine Learning, Computer Vision, Frameworks, Tools.
   - Use \\textbf{{Keyword}} for emphasis within LaTeX, NEVER markdown **.

JOB DESCRIPTION:
{jd}

RESUME LATEX (Full Source):
{resume}

OUTPUT MODIFIED LATEX CODE:"""

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
    "gemini-3-flash-preview": "1. Gemini 3 Flash (20 RPD | The Best, won't break LaTeX)",
    "gemini-2.5-flash": "2. Gemini 2.5 Flash (20 RPD | Highly Capable alternative)",
    "gemini-3.1-flash-lite-preview": "3. Gemini 3.1 Flash Lite (500 RPD | Best for bulk testing)",
    "gemma-3-27b": "4. Gemma 3 27B (14,400 RPD | Massive Backup)"
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
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error("🛑 Rate Limit Exceeded for this Model!")
                    st.warning("💡 **Tip:** You have hit the daily free quota for this specific AI model. Please scroll up and select a different model (e.g. Gemini 3.1 Flash Lite or Gemma) from the dropdown menu to continue!")
                else:
                    st.error(f"An error occurred: {error_msg}")


