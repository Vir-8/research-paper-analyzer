import os
import time
import streamlit as st
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
# Use Gemini 1.5 Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

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

    def compare_papers(self, texts: list) -> str:
        prompt = (
            "You are an expert in comparing and analyzing research papers. "
            "Given the following texts extracted from research papers, provide a comprehensive comparative analysis. "
            "Focus on similarities and differences in their methodology, dataset, results, and future research directions.\n\n"
        )
        for i, text in enumerate(texts, start=1):
            prompt += f"Paper {i} (truncated):\n{text[:4000]}\n\n"
        prompt += "Provide your analysis in a structured, detailed, and easy-to-understand format."
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Comparison Error: {str(e)}"

# Set wide layout for this comparison page
st.set_page_config(page_title="PDF Comparison", layout="wide")

st.title("📑 PDF Comparison")
st.markdown("Upload **2 to 5 research paper PDFs** to generate a comprehensive comparative analysis.")

analyzer = ResearchAnalyzer()

# Initialize a dynamic key for the file uploader if not already set
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = "uploader_1"

# Create a placeholder to hold the file uploader widget
uploader_placeholder = st.empty()

# Reset button to start a new comparison
if st.button("🔄 Start New Comparison"):
    # Update the uploader key with a unique value
    st.session_state["uploader_key"] = f"uploader_{time.time()}"
    # Clear any stored PDF text and comparison data
    for key in list(st.session_state.keys()):
        if key.startswith("pdf_text") or key == "comparison":
            st.session_state.pop(key, None)
    # Reinitialize the uploader by emptying the placeholder
    uploader_placeholder.empty()
    # Optional: if you have access to st.experimental_rerun(), you can uncomment the following line
    # st.experimental_rerun()

# Use the placeholder to render the file uploader with the dynamic key.
pdf_files = uploader_placeholder.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=st.session_state["uploader_key"]
)

if pdf_files:
    if not (2 <= len(pdf_files) <= 5):
        st.error("Please upload between 2 and 5 PDFs.")
    else:
        texts = []
        # Loop over each uploaded file and extract text
        for i, pdf_file in enumerate(pdf_files, start=1):
            key = f"pdf_text{i}"
            with st.spinner(f"Extracting text from PDF {i}..."):
                text = analyzer.extract_text_from_pdf(pdf_file)
                if text.startswith("ERROR"):
                    st.error(f"⚠️ Failed to extract text from PDF {i}. Please try another file.")
                else:
                    st.session_state[key] = text
                    st.success(f"✅ PDF {i} text extracted successfully!")
                    texts.append(text)
        # If texts for all files are available, generate comparative analysis
        if len(texts) == len(pdf_files):
            with st.spinner("Generating comparative analysis..."):
                comparison = analyzer.compare_papers(texts)
                st.session_state.comparison = comparison
            st.markdown(st.session_state.comparison)

st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: #888;">
        Made with ❤️ by <a href="https://github.com/rjm2007" target="_blank">Rudraksh Mehta</a> &amp; 
        <a href="https://github.com/vir-8" target="_blank">Vir Kothari</a>
    </div>
    """,
    unsafe_allow_html=True
)
