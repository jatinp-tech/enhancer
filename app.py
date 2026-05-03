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
PROMPT_TEMPLATE = r"""You are a Grounded Engineering Resume Optimizer. Your mission is to adapt the candidate's resume for a specific Job Description (JD) using direct, plain, and powerful engineering language. Focus on what was built and the measurable impact, avoiding all corporate fluff and "fancy" action verbs.

==== ABSOLUTE CONSTRAINTS (CRITICAL) ====
1. PAGE LIMIT & CONTENT RETENTION: The output MUST stay on one page. HOWEVER, you MUST preserve the approximate length, detail, and technical depth of the original resume. Do NOT over-summarize or aggressively cut content. Do NOT reduce content density.
2. NO SECTION REMOVAL: Do NOT remove any major sections (Summary, Work Experience, Projects, Publications, Skills, Education). ALL sections must remain in the output.
3. BULLET POINT COUNT: Keep the EXACT SAME number of bullet points per role/project. Do NOT add new ones, and do NOT delete existing ones. Maintain similar length and technical depth for each bullet. Do NOT merge or split bullets. SUMMARY must be max 3 lines.
4. RAW LATEX ONLY: Output ONLY pure LaTeX code. 
   - ✗ NO MARKDOWN BOLD (Do NOT use **keyword**)
   - ⚠️ YOUR ENTIRE OUTPUT IS INVALID IF IT CONTAINS ANY DOUBLE ASTERISKS (**).
5. PRESERVE STRUCTURE: 
   - Keep all LaTeX commands (\newcommand, \usepackage, \geometry, \vspace, \item) EXACTLY intact.
   - DO NOT modify LaTeX syntax, commands, brackets, or structure.
6. STRICTLY FORBIDDEN (HONESTY ENFORCEMENT): Do NOT include: Kubernetes, Terraform, CI/CD, Cloud Platforms (AWS, Azure, GCP), "Advanced Pipelines", "Agentic AI", "R", "R language", "large scale", "millions of users", or "high-traffic" unless explicitly present in the original resume. Explicitly ignore these. Do NOT add irrelevant information the candidate does not have, such as embedded hardware programming or perceptron systems.
7. NO TITLE CHANGE (CRITICAL): The candidate's title MUST remain "Machine Learning Engineer" (or whatever is in the original). Do NOT add "Data Scientist", "AI Architect", "Lead", or any other variations in the Summary or headers.
8. NO HALLUCINATION/FABRICATION: Do NOT invent, rename, or substitute ANY project, job, or experience. Project titles MUST be copied VERBATIM from the original resume. You may only rephrase bullet descriptions — never the title itself.
9. LAYOUT PRESERVATION: Do NOT change spacing, formatting, or line structure that could affect the one-page layout.
10. BULLET COUNT ENFORCEMENT: Count bullets per role/project in the original before writing. The output MUST have the EXACT same count. Do NOT silently drop or merge any bullet.
11. METRICS PRESERVATION: You MUST strictly retain ALL numbers, percentages, timeframes, and quantifiable metrics from the original resume. Do NOT drop or paraphrase integers/numbers out of the bullet points.
12. SKILL GAP ANALYSIS ("MARK TO LEARN"): At the very end of your LaTeX output, after \end{{document}}, add a LaTeX comment block starting with `% MISSING SKILLS TO LEARN:` followed by a comma-separated list of key skills required by the JD that the candidate currently lacks. This helps the candidate know what to learn.

==== CANDIDATE EXPERTISE SOURCE OF TRUTH ====
- ML CORE: Python, Model Evaluation, Data Pipelines, Supervised Learning, Metric Learning.
- COMPUTER VISION (High-Tier): YOLOv8, ConvNeXt, OpenCV, DINO/CLIP embeddings, Object shape-based detection, Image preprocessing.
- INFERENCE & DEPLOYMENT: FastAPI, Streamlit, GPU batching, Latency optimization (25-40% reduction), local GPU servers.
- ROBUSTNESS & SECURITY: Adversarial learning (FGSM, PGD), anomaly detection (Isolation Forest), cryptographic analysis.
- NLP (Basic): Text classification, sentiment analysis fundamentals; familiar with Transformer architecture concepts (BERT, RoBERTa).
- GEN AI/LLMS: Basic RAG, OpenAI text-embedding-3, GPT-4, Neo4j (Search integration).
- TOOLS/DBs: SQL, Git, Linux, Jupyter.

==== OPTIMIZATION STRATEGY (GROUNDED & DIRECT) ====
1. GROUNDED LANGUAGE (CRITICAL):
   - The resume MUST use direct, plain engineering language. Avoid flowery "storytelling" or corporate fluff.
   - Focus on the "Action -> Result": Clearly state the specific engineering task, the tool used, and the quantifiable outcome.
   - FORBIDDEN FANCY VERBS: Do NOT use "Architected", "Spearheaded", "Orchestrated", "Leveraged", "Pioneered", "Harnessed", "Conceptualized", "Transformed", or "Championed".
   - PREFERRED VERBS: Use Built, Developed, Implemented, Improved, Reduced, Scaled, Optimized, Trained, Deployed, or Integrated.
2. SUMMARY AS A DIRECT HOOK:
   - Write a concise summary that highlights the candidate as a "Machine Learning Engineer" with specific strengths.
   - No fluff. No "driving business value through...". Instead: "Machine Learning Engineer with 3 years of experience specializing in CV and ML Systems..."
   - Do NOT add \textbf{{}} bolding inside the Summary. Plain text only.
3. EXPERIENCE BULLETS (ENGINEERING-FIRST):
   - Restructure bullets to follow a clear "Built [X] using [Y] to achieve [Z]" flow. 
   - NO FORCE-MATCHING (CRITICAL): Do NOT change the technical substance, tools used, or scope of the original work to match the JD. Only use JD keywords if they were actually part of the original project. Do NOT hallucinate skills into a project where they didn't exist.
   - METRIC-DRIVEN IMPACT: The "Result" must explicitly highlight the numbers, percentages, and metrics from the original resume. Never remove an integer or quantifiable metric.
   - AVOID ROBOTIC OR FLOWERY TONE: Ensure the tone is human but professional and direct.
   - FORBIDDEN FILLER PHRASES: Do NOT use: "Technical Excellence", "Investigations", "Data Insights", "Compliance", "Integrity", "Forensics", "Business Value", "Compelling Narrative".
4. RESEARCH & PROJECTS:
   - Align bullet descriptions to tell the story of the project's goals, the methods used, and the final outcomes. Project titles MUST remain VERBATIM.
5. SKILLS SECTION (ATS KEYWORD ENGINE):
   - This section is your primary tool for ATS optimization.
   - Inject the EXACT keywords from the Job Description here, provided they align with the candidate's "Source of Truth".
   - Use ONLY these EXACT category headers: Programming, Machine Learning, Deep Learning, ML Systems, Frameworks / Libraries, Tools.
   - Maintain the existing LaTeX formatting (e.g., \textbf{{Category:}}). Do NOT add extra bolding.

JOB DESCRIPTION:
{jd}

RESUME LATEX (Full Source):
{resume}
OUTPUT MODIFIED LATEX CODE:"""

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
