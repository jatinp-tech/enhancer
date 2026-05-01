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
PROMPT_TEMPLATE = r"""You are an ATS Resume Optimizer (like Jobscan). Rewrite the candidate's LaTeX resume for the given Job Description. Output ONLY raw LaTeX — no markdown, no explanation.

=== HARD RULES ===
1. Output ONLY valid LaTeX. No **, no markdown, no triple backticks.
2. Keep ALL sections (Summary, Work Experience, Projects, Publications, Skills, Education).
3. Keep the EXACT same number of bullet points per role. Do NOT add, merge, drop, or split bullets.
4. Keep ALL numbers, percentages, and metrics verbatim. Never remove a quantifiable result.
5. Do NOT change the candidate's job title. Do NOT rename projects — copy titles verbatim.
6. Do NOT hallucinate: no Kubernetes, Docker, Terraform, CI/CD, AWS/Azure/GCP, R, embedded hardware, "large-scale", "millions of users" unless already in the original.
7. Preserve ALL LaTeX commands (\newcommand, \usepackage, \geometry, \vspace, \item) exactly. Do not change spacing or layout.
8.SUMMARY AS A HOOK:
   - Write an engaging opening narrative that frames the candidate as an effective problem solver.
   - You MAY naturally weave 2-3 JD keywords into the summary, but ONLY if they reflect real skills and fit the narrative naturally.
9. Skills section: use ONLY these headers: Programming, Machine Learning, Deep Learning, ML Systems, Frameworks / Libraries, Tools. Inject JD keywords that match the candidate's real skills.

=== CANDIDATE SKILLS (Source of Truth) ===
ML: Python, Supervised Learning, Clustering, PCA, XGBoost, SVM, Metric Learning
Vision: YOLOv8, ConvNeXt, OpenCV, DINO/CLIP, ArcFace, object shape-based detection
Deployment: FastAPI, Streamlit, GPU batching, latency optimization (25-40% reduction), local GPU servers
Robustness: Adversarial ML (FGSM, PGD) in CNN, Isolation Forest anomaly detection, cryptographic analysis
GenAI: RAG basics, Text embeddings, Bert
Tools: SQL, Git, Linux, Jupyter

=== WRITING STYLE ===
- Each bullet: "Action → Challenge/Context → Result". Keep tone professional and human.
- Tailor emphasis to the JD (MLOps → deployment; Security → robustness; Vision → pipelines).
- Do NOT use filler phrases: "Technical Excellence", "Data Insights", "Compliance", "Integrity".

JOB DESCRIPTION:
{jd}

RESUME LATEX:
{resume}
OUTPUT:"""

def clean_markdown(text: str) -> str:
    for artifact in MARKDOWN_ARTIFACTS:
        text = text.replace(artifact, "")
    return text.strip()

def sanitize_latex(text: str) -> str:
    # LLMs often generate unicode characters that crash pdflatex or ATS text extractors.
    # We replace them with safe LaTeX ASCII equivalents before compiling.
    replacements = {
        "“": "``", "”": "''", "‘": "`", "’": "'",
        "—": "---", "–": "--", "…": "...", "•": "\\textbullet{}",
        " ": " ", " ": " ", "​": "", # Non-breaking spaces and zero-width spaces
        "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", 
        # Wait, if we replace &, %, $, #, we might break actual LaTeX commands!
        # Gemini is outputting LaTeX, so it *should* already escape them.
        # But we MUST fix the unicode quotes and dashes!
    }
    # Safely replace only the known bad unicode quotes and spaces
    safe_replacements = {
        "“": "``", "”": "''", "‘": "`", "’": "'",
        "—": "---", "–": "--", "…": "...", "•": "\\textbullet{}",
        " ": " ", " ": " ", "​": ""
    }
    for old, new in safe_replacements.items():
        text = text.replace(old, new)
    return text

def optimize_resume(jd: str, resume: str, model_id: str) -> str:
    client = genai.Client(api_key=API_KEY)
    prompt = PROMPT_TEMPLATE.format(jd=jd, resume=resume)
    response = client.models.generate_content(model=model_id, contents=prompt)
    cleaned_text = clean_markdown(response.text)
    return sanitize_latex(cleaned_text)

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
                        
                        if compile_process.returncode == 0 and os.path.exists(pdf_path):
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
                            st.error("❌ Failed to compile LaTeX to PDF. The model likely generated invalid LaTeX structure or there is a syntax error.")
                            with st.expander("View LaTeX Errors"):
                                st.text(compile_process.stdout)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error("🛑 Rate Limit Exceeded for this Model!")
                    st.warning("💡 **Tip:** You have hit the daily free quota for this specific AI model. Please scroll up and select a different model (e.g. Gemini 3.1 Flash Lite or Gemma) from the dropdown menu to continue!")
                else:
                    st.error(f"An error occurred: {error_msg}")


