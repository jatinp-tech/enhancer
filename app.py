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
PROMPT_TEMPLATE = r"""You are an elite Resume ATS Optimizer. Rewrite the candidate's LaTeX resume for the given Job Description. Output ONLY pure LaTeX.

=== CRITICAL RULES ===
1. Output ONLY valid LaTeX. No markdown, no **bolding**, no triple backticks.
2. Keep ALL sections: Summary, Work Experience, Projects, Publications, Skills, Education.
3. Keep the EXACT same number of bullet points per section. Do NOT add/drop/merge bullets.
4. Keep ALL numbers, percentages, and metrics verbatim.
5. Do NOT change job titles or project titles. Copy them verbatim.
6. PRESERVE STRUCTURE: Keep all commands (\newcommand, \usepackage, \geometry, \vspace, \item, \resumeItemListStart, \resumeEducation) exactly as they are. Do NOT substitute commands.
7. SUMMARY AS A HOOK:
   - Write an engaging opening narrative (max 3 lines) that frames the candidate as an effective problem solver.
   - You MAY naturally weave 2-3 JD keywords into the summary, but ONLY if they reflect real skills and fit the narrative naturally.
   - Do NOT use \textbf{{}} bolding inside the summary.
8. FORBIDDEN WORDS: Do NOT use "Technical Excellence", "Investigations", "Data Insights", "Compliance", "Integrity", "Forensics", "Results-driven".

=== CANDIDATE TRUTH (Source of Truth) ===
- ML: Python, Model Evaluation, Data Pipelines, Supervised Learning, Metric Learning.
- VISION: YOLOv8, ConvNeXt, OpenCV, DINO/CLIP, Object shape detection.
- DEPLOYMENT: FastAPI, Streamlit, GPU batching, Latency optimization (25-40%).
- SECURITY: Adversarial learning (FGSM, PGD), Isolation Forest anomaly detection.
- GEN AI/NLP: RAG, Transformers (BERT, RoBERTa), Text Embeddings.

=== WRITING STYLE ===
- Bullet points: "Action → Context → Result". Tell a story of impact.
- Tailor focus based on JD (MLOps vs Security vs Vision).
- Keep tone human, engineering-focused, and professional.

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
    # The PROMPT_TEMPLATE uses doubled braces for \textbf{{}} etc. to work with .format()
    prompt = PROMPT_TEMPLATE.format(jd=jd, resume=resume)
    response = client.models.generate_content(model=model_id, contents=prompt)
    cleaned_text = clean_markdown(response.text)
    return sanitize_latex(cleaned_text)

def verify_prompt(prompt: str) -> list:
    """Simple validation of the generated LaTeX prompt.
    Returns a list of error messages (empty if ok)."""
    errors = []
    # 1. No double asterisks (markdown bold)
    if "**" in prompt:
        errors.append("Found markdown bold (**). Must be removed.")
    # 2. Forbidden terms list
    forbidden = ["Technical Excellence", "Investigations", "Data Insights", "Compliance", "Integrity", "Forensics", "Results-driven"]
    for term in forbidden:
        if term.lower() in prompt.lower():
            errors.append(f"Forbidden term detected: {term}")
    # 3. Required sections
    required_sections = ["Summary", "Work Experience", "Projects", "Publications", "Skills", "Education"]
    for sec in required_sections:
        if f"\\section{{\\textbf{{{sec}}}}}" not in prompt:
            errors.append(f"Missing required section: {sec}")
    # 4. Ensure no unbalanced lists
    if prompt.count("\\resumeItemListStart") != prompt.count("\\resumeItemListEnd"):
        errors.append("Unbalanced resumeItemList (Start vs End mismatch).")
    return errors

# --- STREAMLIT UI ---
st.set_page_config(page_title="LaTeX Resume Optimizer", page_icon="📄")
st.title("📄 LaTeX Resume Optimizer")

st.markdown("ATS resume optimizer with JD matching using Google AI Studio selected models.")

MODEL_OPTIONS = {
    "gemini-2.0-flash": "1. Gemini 2.0 Flash (20 RPD | The Best)",
    "gemini-1.5-flash": "2. Gemini 1.5 Flash (20 RPD)",
    "gemini-1.5-flash-8b": "3. Gemini 1.5 Flash 8B (500 RPD)",
    "gemma-2-27b": "4. Gemma 2 27B (Backup)"
}

selected_model = st.selectbox("Choose Model:", options=list(MODEL_OPTIONS.keys()), format_func=lambda x: MODEL_OPTIONS[x])
jd_input = st.text_area("Paste JD here:", height=300)

if st.button("Generate Optimized Resume", type="primary"):
    if not jd_input:
        st.warning("⚠️ Please provide a Job Description.")
    else:
        with st.spinner("Optimizing your resume with Gemini..."):
            try:
                with open("resume.tex", "r", encoding="utf-8") as f:
                    resume_content = f.read()
            except FileNotFoundError:
                st.error("❌ 'resume.tex' was not found in the repository!")
                st.stop()
            try:
                optimized_tex = optimize_resume(jd_input, resume_content, selected_model)
                
                # Verify the generated LaTeX
                v_errors = verify_prompt(optimized_tex)
                if v_errors:
                    st.error("🔍 Prompt verification failed:")
                    for err in v_errors: st.write(f"- {err}")
                    st.stop()
                
                st.success(f"✅ Optimization complete using {selected_model}!")
                
                # Save and offer .tex download
                with open("optimized.tex", "w", encoding="utf-8") as f:
                    f.write(optimized_tex)
                st.download_button("⬇️ Download Optimized LaTeX (.tex)", optimized_tex, "optimized.tex")
                
                with st.spinner("Compiling LaTeX to PDF..."):
                    import tempfile, subprocess, os
                    with tempfile.TemporaryDirectory() as td:
                        tp = os.path.join(td, "optimized.tex")
                        pp = os.path.join(td, "optimized.pdf")
                        lp = os.path.join(td, "optimized.log")
                        
                        with open(tp, "w", encoding="utf-8") as f: f.write(optimized_tex)
                        
                        pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "optimized.tex"]
                        
                        # Double-pass compilation for ResumeGo compatibility (resolves cross-refs/outlines)
                        subprocess.run(pdflatex_cmd, cwd=td, capture_output=True)
                        compile_process = subprocess.run(pdflatex_cmd, cwd=td, capture_output=True, text=True)
                        
                        has_fatal = False
                        if os.path.exists(lp):
                            with open(lp, "r", errors="ignore") as lf:
                                log_text = lf.read()
                                if "Fatal error" in log_text or "Emergency stop" in log_text:
                                    has_fatal = True
                        
                        if compile_process.returncode == 0 and os.path.exists(pp) and not has_fatal:
                            with open(pp, "rb") as f:
                                st.success("🎉 PDF Compiled Successfully!")
                                st.download_button("⬇️ Download Optimized PDF", f.read(), "optimized.pdf", "application/pdf")
                        else:
                            st.warning("⚠️ PDF compilation failed. Download the .tex and compile locally.")
                            with st.expander("View LaTeX Errors"):
                                st.text(compile_process.stdout[-3000:] if compile_process.stdout else "No output")
            except Exception as e:
                st.error(f"An error occurred: {e}")
