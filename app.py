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
PROMPT_TEMPLATE = r"""You are an elite LaTeX Resume Optimizer and Storyteller. Your mission is to adapt the candidate's resume for a specific Job Description (JD) by crafting a compelling narrative of their engineering impact, rather than just stuffing it with professional keywords.

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
7. NO TITLE CHANGE: Do NOT change the candidate's existing job profile title/role. It MUST remain EXACTLY as it is in the original resume.
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

==== OPTIMIZATION STRATEGY (NARRATIVE & STORYTELLING) ====
1. STORYTELLING OVER KEYWORDS (CRITICAL):
   - The resume MUST read like a compelling narrative of impact, problem-solving, and engineering excellence, NOT a robotic list of ATS keywords.
   - Focus on the "Why" and "How": When rewriting bullets, clarify the core engineering challenge, the specific approach taken, and the quantifiable result.
   - Do NOT awkwardly shoehorn JD keywords. If a required tool aligns with their "Source of Truth", integrate it smoothly into the story of what was built. If it breaks the flow, leave it out.
2. SUMMARY AS A HOOK:
   - Write an engaging opening narrative that frames the candidate as an effective problem solver.
   - You MAY naturally weave 2-3 JD keywords into the summary, but ONLY if they reflect real skills and fit the narrative naturally.
   - Do NOT add \textbf{{}} bolding inside the Summary. Plain text only.
3. EXPERIENCE BULLETS (ACTION-IMPACT NARRATIVE):
   - Restructure bullets to follow a clear "Action -> Context/Challenge -> Result" flow. Tell a mini-story in each bullet.
   - METRIC-DRIVEN IMPACT: The "Result" must explicitly highlight the numbers, percentages, and metrics from the original resume. Never remove an integer or quantifiable metric when rewriting.
   - DYNAMIC MAPPING: Tailor the narrative to the JD. For MLOps, tell the story of their inference and deployment optimization. For AI Security, tell the story of their robustness and adversarial defenses. For Vision, highlight their deep learning pipeline challenges.
   - AVOID ROBOTIC TONE: Ensure the tone is engaging, human, and professional. Remove any phrasing that sounds artificially generated or "ATS-optimized".
   - FORBIDDEN FILLER PHRASES: Do NOT use: "Technical Excellence", "Investigations", "Data Insights", "Compliance", "Integrity", "Forensics".
4. RESEARCH & PROJECTS:
   - Align bullet descriptions to tell the story of the project's goals, the innovative methods used, and the final outcomes. Project titles MUST remain VERBATIM.
5. SKILLS SECTION (ATS KEYWORD ENGINE):
   - This section is your primary tool for ATS optimization.
   - Inject the EXACT keywords from the Job Description here, provided they align with the candidate's "Source of Truth". (e.g., if JD asks for "Object Detection", add it here alongside YOLOv8).
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
                
                # Always save and offer the .tex file for download (works even on Streamlit Cloud without pdflatex)
                with open("optimized.tex", "w", encoding="utf-8") as f:
                    f.write(optimized_tex)
                st.download_button(
                    label="⬇️ Download Optimized LaTeX (.tex)",
                    data=optimized_tex.encode("utf-8"),
                    file_name="optimized.tex",
                    mime="text/plain"
                )
                
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
                            st.warning("⚠️ pdflatex not available on this server. Use the .tex download above and compile locally with: `pdflatex optimized.tex`")
                            with st.expander("View LaTeX Compilation Errors"):
                                st.text(compile_process.stdout)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    st.error("🛑 Rate Limit Exceeded for this Model!")
                    st.warning("💡 **Tip:** You have hit the daily free quota for this specific AI model. Please scroll up and select a different model (e.g. Gemini 3.1 Flash Lite or Gemma) from the dropdown menu to continue!")
                else:
                    st.error(f"An error occurred: {error_msg}")


