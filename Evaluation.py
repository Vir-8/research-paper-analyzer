import os
import streamlit as st
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
import spacy

# Load environment variables
load_dotenv()

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("⚠️ Please set your GEMINI_API_KEY in the environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)
# Use Gemini 1.5 Flash instead of gemini-pro
model = genai.GenerativeModel('gemini-1.5-flash')

@dataclass
class Dataset:
    source: str = "Not specified"
    size: str = "Not specified"
    data_type: str = "Not specified"
    processing_steps: List[str] = field(default_factory=list)

@dataclass
class Methodology:
    core_approach: str = "Not specified"
    techniques: List[str] = field(default_factory=list)
    novelty: Optional[str] = None

@dataclass
class Results:
    quantitative: Dict[str, Any] = field(default_factory=dict)
    qualitative: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=list)

@dataclass
class FutureDirections:
    author_stated: List[str] = field(default_factory=list)
    implied_gaps: List[str] = field(default_factory=list)

@dataclass
class PaperAnalysis:
    title: str
    year: int
    url: Optional[str]
    methodology: Methodology
    dataset: Dataset
    results: Results
    future_directions: FutureDirections
    confidence_score: float
    missing_sections: List[str]
    
    def to_markdown(self) -> str:
        md = [
            f"# {self.title} ({self.year})",
            f"**Link**: {self.url or 'Not found'}\n",
            "## Methodology",
            f"* Core approach: {self.methodology.core_approach}",
            "* Techniques:",
            *[f"  - {tech}" for tech in self.methodology.techniques],
            f"* Novelty: {self.methodology.novelty or 'Not specified'}\n",
            "## Dataset",
            f"* Source: {self.dataset.source}",
            f"* Size/Type: {self.dataset.size} | {self.dataset.data_type}",
            "* Processing:",
            *[f"  - {step}" for step in self.dataset.processing_steps],
            "\n## Results",
            "* Quantitative:",
            *[f"  - {k}: {v}" for k, v in self.results.quantitative.items()],
            "* Qualitative:",
            *[f"  - {obs}" for obs in self.results.qualitative],
            "* Benchmarks:",
            *[f"  - {bench}" for bench in self.results.benchmarks],
            "\n## Future Directions",
            "* Author-stated:",
            *[f"  - {gap}" for gap in self.future_directions.author_stated],
            "* Implied gaps:",
            *[f"  - {gap}" for gap in self.future_directions.implied_gaps],
            f"\n**Confidence Score**: {self.confidence_score:.2f}%\n",
            "## Missing Data",
            *[f"- {section}" for section in self.missing_sections]
        ]
        return "\n".join(md)

class ResearchAnalyzer:
    def __init__(self):
        self.nlp = nlp
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text if text else "ERROR: No extractable text found in PDF."
        except Exception as e:
            return f"ERROR: {str(e)}"

    def analyze_paper(self, text: str) -> str:
        prompt = f"""
        You are an expert in summarizing and analyzing research papers. Based on the text extracted from a research paper, generate a detailed and structured literature review. Follow this format:

        **Literature Review Structure**
        - **Title of the Paper:** (Extract the title)
        - **Year of the Paper:** (Extract the publication year)
        - **Methodology:** (Summarize the methodology used)
        - **Dataset:** (Describe dataset details including size and source)
        - **Results:** (Highlight key results)
        - **Future Work/Research Gaps:** (Identify proposed future work)
        - **Insights:** (Provide additional observations)
        - **Missing Sections:** (If any sections are missing, state them clearly)

        **Paper text (truncated for analysis):**
        {text[:8000]}
        """
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI Analysis Error: {str(e)}"

# Set wide layout for main content
st.set_page_config(page_title="Research Paper Analyzer", layout="wide")

# Custom CSS to move the sidebar to the right side
st.markdown(
    """
    <style>
    /* Move Streamlit sidebar to the right */
    [data-testid="stSidebar"] {
        left: auto;
        right: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main page title and description
st.title("📄 Research Paper Analyzer")
st.markdown("Upload a **research paper PDF** to get both an AI-generated literature review and an interactive chatbot for queries.")

# File uploader for the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    if "pdf_text" not in st.session_state:
        with st.spinner("Extracting text from PDF..."):
            analyzer = ResearchAnalyzer()
            pdf_text = analyzer.extract_text_from_pdf(uploaded_file)
            if pdf_text.startswith("ERROR"):
                st.error("⚠️ Failed to extract text from the PDF. Please try another file.")
            else:
                st.session_state.pdf_text = pdf_text
                st.success("✅ Text extracted successfully!")
    
    if "analysis" not in st.session_state and "pdf_text" in st.session_state:
        with st.spinner("Generating AI analysis..."):
            analyzer = ResearchAnalyzer()
            analysis = analyzer.analyze_paper(st.session_state.pdf_text)
            st.session_state.analysis = analysis

    # Layout: Left column for literature review summary
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📑 Literature Review Summary")
        st.markdown(st.session_state.analysis)
        if st.button("📥 Download as Markdown"):
            markdown_path = Path("analysis_report.md")
            markdown_path.write_text(st.session_state.analysis)
            st.download_button("Download Summary", markdown_path.read_bytes(), "analysis_report.md", "text/markdown")

    # Chatbot now in the sidebar on the right (using our CSS hack)
    with st.sidebar:
        st.header("🤖 Chatbot")
        # Render the Clear Chat button at the top
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
        
        # Chat input form for new messages
        with st.form(key="chat_form_sidebar", clear_on_submit=True):
            user_question = st.text_area("Your question:", height=100)
            submitted = st.form_submit_button("Send")
            if submitted and user_question:
                with st.spinner("Generating answer..."):
                    chat_prompt = f"""
                    You are a knowledgeable research assistant with expertise in academic papers.
                    Below is an AI-generated literature review extracted from the paper.
                    Use it as context, but feel free to draw on your broader expertise to provide a comprehensive answer.
                    
                    Context:
                    {st.session_state.analysis}
                    
                    Question:
                    {user_question}
                    """
                    try:
                        response = model.generate_content(chat_prompt)
                        answer = response.text
                    except Exception as e:
                        answer = f"Error generating answer: {str(e)}"
                
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "bot", "content": answer})
        
        st.markdown("---")
        st.subheader("Conversation")
        # Display the conversation with styled message bubbles
        if "chat_history" in st.session_state:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f"<div style='background-color:#cce5ff; color:#000; padding:10px; border-radius:8px; margin:8px 0;'>"
                        f"<strong>User:</strong> {msg['content']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:#e2e3e5; color:#000; padding:10px; border-radius:8px; margin:8px 0;'>"
                        f"<strong>Bot:</strong> {msg['content']}</div>",
                        unsafe_allow_html=True
                    )
