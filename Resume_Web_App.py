from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Environment Variables ---
load_dotenv()

# --- Load the Sentence Transformer Model ---
@st.cache_resource
def load_model():
    # Using a lightweight, effective model for semantic embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

# --- Function to Extract Text from PDF ---
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

# --- Function to Compute ATS Scores using Sentence Embeddings ---
def get_ats_scores(job_description, resumes):
    """
    Computes the absolute semantic similarity between the job description and each resume,
    and produces an ATS score by multiplying the cosine similarity by 100.

    This score reflects the alignment of each resume with the job description without
    scaling based on the min or max values among resumes.
    """
    # Combine the job description and resumes into a single list for embedding
    texts = [job_description] + resumes
    embeddings = model.encode(texts)

    # The first embedding is for the job description
    jd_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]

    # Compute cosine similarities between job description and each resume
    cosine_scores = cosine_similarity([jd_embedding], resume_embeddings).flatten()

    # Multiply cosine similarity (0 to 1) by 100 to get an ATS score
    ats_scores = cosine_scores * 100
    return ats_scores

# --- Streamlit Application ---
st.title("AI Resume Screening & Candidate Ranking System")

st.header("Job Description")
job_description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes_text = []
    resume_names = []
    
    # Extract text from each uploaded PDF
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes_text.append(text)
        resume_names.append(file.name)
    
    # Calculate ATS scores using semantic embeddings (absolute alignment score)
    ats_scores = get_ats_scores(job_description, resumes_text)
    
    # Create a DataFrame to display the results
    results = pd.DataFrame({
        "Resume": resume_names,
        "ATS Score": ats_scores
    }).sort_values(by="ATS Score", ascending=False)
    
    st.write("### Resume Ranking Results")
    st.dataframe(results)
    
    # Visualization using a Plotly bar chart
    fig = px.bar(results, x="Resume", y="ATS Score", 
                 title="Resume ATS Scores (0-100)",
                 labels={"ATS Score": "ATS Score (0-100)"})
    st.plotly_chart(fig)
    
    # Download button to export results as CSV
    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="resume_ranking_results.csv",
        mime="text/csv"
    )

