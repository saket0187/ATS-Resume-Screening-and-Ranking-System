from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# --- Load Environment Variables ---
load_dotenv()

# --- Load the Sentence Transformer Model ---
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

# --- Text Preprocessing Functions ---
def preprocess_text(text):
    """Clean and preprocess text for better analysis"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# --- Function to Extract Text from PDF ---
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

# --- Keyword Extraction and Matching ---
def extract_keywords(text, num_keywords=50):
    """Extract important keywords using TF-IDF"""
    preprocessed_text = preprocess_text(text)
    
    # Use TF-IDF to extract keywords
    vectorizer = TfidfVectorizer(max_features=num_keywords, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Get top keywords
    keyword_scores = dict(zip(feature_names, tfidf_scores))
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_keywords

def keyword_matching_score(jd_keywords, resume_text):
    """Calculate keyword matching score"""
    resume_text_lower = resume_text.lower()
    preprocessed_resume = preprocess_text(resume_text)
    
    matched_keywords = 0
    total_weight = 0
    
    for keyword, weight in jd_keywords:
        total_weight += weight
        if keyword in preprocessed_resume:
            matched_keywords += weight
    
    return (matched_keywords / total_weight * 100) if total_weight > 0 else 0

# --- Semantic Similarity Functions ---
def semantic_similarity_score(job_description, resume_text):
    """Calculate semantic similarity using sentence transformers"""
    embeddings = model.encode([job_description, resume_text])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100

def tfidf_similarity_score(job_description, resume_text):
    """Calculate TF-IDF based similarity"""
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100

# --- Experience and Skills Extraction ---
def extract_experience_years(text):
    """Extract years of experience from resume text"""
    # Common patterns for experience
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in\s*\w+',
    ]
    
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        years.extend([int(year) for year in matches])
    
    return max(years) if years else 0

def extract_skills(text, skill_keywords):
    """Extract skills mentioned in the resume"""
    text_lower = text.lower()
    found_skills = []
    
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return found_skills

# --- Composite Scoring Algorithm ---
def calculate_composite_score(job_description, resume_text, resume_name):
    """Calculate a comprehensive score using multiple algorithms"""
    
    # 1. Keyword Matching Score (30% weight)
    jd_keywords = extract_keywords(job_description, 30)
    keyword_score = keyword_matching_score(jd_keywords, resume_text)
    
    # 2. Semantic Similarity Score (35% weight)
    semantic_score = semantic_similarity_score(job_description, resume_text)
    
    # 3. TF-IDF Similarity Score (20% weight)
    tfidf_score = tfidf_similarity_score(job_description, resume_text)
    
    # 4. Experience Score (15% weight)
    experience_years = extract_experience_years(resume_text)
    experience_score = min(experience_years * 10, 100)  # Cap at 100
    
    # Weighted composite score
    composite_score = (
        keyword_score * 0.30 +
        semantic_score * 0.35 +
        tfidf_score * 0.20 +
        experience_score * 0.15
    )
    
    return {
        'composite_score': composite_score,
        'keyword_score': keyword_score,
        'semantic_score': semantic_score,
        'tfidf_score': tfidf_score,
        'experience_score': experience_score,
        'experience_years': experience_years
    }

# --- Personalized Suggestions Function ---
def generate_personalized_suggestions(job_description, resume_text, resume_name):
    """
    Generate personalized suggestions to improve ATS score
    """
    suggestions = []
    
    # Extract keywords from job description
    jd_keywords = extract_keywords(job_description, 20)
    top_jd_keywords = [keyword for keyword, score in jd_keywords[:15]]
    
    # Extract keywords from resume
    resume_keywords = extract_keywords(resume_text, 20)
    resume_keyword_list = [keyword for keyword, score in resume_keywords]
    
    # Find missing important keywords
    missing_keywords = []
    for keyword in top_jd_keywords:
        if keyword not in resume_text.lower():
            missing_keywords.append(keyword)
    
    # Keyword suggestions
    if missing_keywords:
        suggestions.append({
            'category': 'Keywords',
            'priority': 'High',
            'suggestion': f"Add these important keywords: {', '.join(missing_keywords[:5])}",
            'impact': 'Could increase keyword matching score by 15-25%'
        })
    
    # Experience suggestions
    experience_years = extract_experience_years(resume_text)
    jd_experience_years = extract_experience_years(job_description)
    
    if jd_experience_years > 0 and experience_years == 0:
        suggestions.append({
            'category': 'Experience',
            'priority': 'High',
            'suggestion': 'Clearly mention your years of experience (e.g., "5+ years of experience in...")',
            'impact': 'Could increase experience score significantly'
        })
    elif jd_experience_years > experience_years and experience_years > 0:
        suggestions.append({
            'category': 'Experience',
            'priority': 'Medium',
            'suggestion': f'Consider highlighting projects or skills that demonstrate equivalent experience to the required {jd_experience_years} years',
            'impact': 'Could improve experience relevance'
        })
    
    # Skills suggestions
    common_tech_skills = ['python', 'java', 'sql', 'javascript', 'machine learning', 'data analysis', 
                         'project management', 'agile', 'scrum', 'git', 'aws', 'docker', 'kubernetes']
    
    jd_lower = job_description.lower()
    resume_lower = resume_text.lower()
    
    missing_tech_skills = []
    for skill in common_tech_skills:
        if skill in jd_lower and skill not in resume_lower:
            missing_tech_skills.append(skill)
    
    if missing_tech_skills:
        suggestions.append({
            'category': 'Technical Skills',
            'priority': 'High',
            'suggestion': f"Add these technical skills if you have them: {', '.join(missing_tech_skills[:3])}",
            'impact': 'Could increase semantic similarity score by 10-20%'
        })
    
    # Format suggestions
    if 'responsibilities' in jd_lower and 'responsible' not in resume_lower:
        suggestions.append({
            'category': 'Content Format',
            'priority': 'Medium',
            'suggestion': 'Use action-oriented language like "Responsible for", "Managed", "Developed", "Implemented"',
            'impact': 'Could improve semantic matching'
        })
    
    # Quantifiable achievements
    numbers_in_resume = len(re.findall(r'\d+', resume_text))
    if numbers_in_resume < 5:
        suggestions.append({
            'category': 'Quantifiable Results',
            'priority': 'Medium',
            'suggestion': 'Add quantifiable achievements (e.g., "Increased efficiency by 30%", "Managed team of 5 people")',
            'impact': 'Could improve overall relevance and impact'
        })
    
    # Industry-specific terms
    industry_terms = extract_industry_terms(job_description)
    missing_industry_terms = []
    for term in industry_terms:
        if term not in resume_lower:
            missing_industry_terms.append(term)
    
    if missing_industry_terms:
        suggestions.append({
            'category': 'Industry Terms',
            'priority': 'Medium',
            'suggestion': f"Consider adding industry-specific terms: {', '.join(missing_industry_terms[:3])}",
            'impact': 'Could improve industry relevance'
        })
    
    return suggestions

def extract_industry_terms(job_description):
    """Extract potential industry-specific terms"""
    # Common industry terms and patterns
    industry_patterns = [
        r'(fintech|healthcare|e-commerce|saas|blockchain|ai|ml|data science)',
        r'(retail|banking|insurance|telecommunications|manufacturing)',
        r'(startup|enterprise|consulting|agency|non-profit)'
    ]
    
    terms = []
    text_lower = job_description.lower()
    
    for pattern in industry_patterns:
        matches = re.findall(pattern, text_lower)
        terms.extend(matches)
    
    return list(set(terms))

# --- Enhanced ATS Scoring Function ---
def get_enhanced_ats_scores(job_description, resumes, resume_names):
    """
    Compute comprehensive ATS scores using multiple algorithms
    """
    results = []
    
    for i, resume_text in enumerate(resumes):
        scores = calculate_composite_score(job_description, resume_text, resume_names[i])
        scores['resume_name'] = resume_names[i]
        
        # Generate personalized suggestions
        suggestions = generate_personalized_suggestions(job_description, resume_text, resume_names[i])
        scores['suggestions'] = suggestions
        
        results.append(scores)
    
    return results

# --- Streamlit Application ---
st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("🎯 AI Resume Screening & Candidate Ranking System")
st.markdown("*Advanced multi-algorithm approach for accurate candidate assessment*")

# Sidebar for configuration
st.sidebar.header("Configuration")
show_detailed_scores = st.sidebar.checkbox("Show Detailed Scores", value=True)
min_score_threshold = st.sidebar.slider("Minimum Score Threshold", 0, 100, 50)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Job Description")
    job_description = st.text_area("Enter the job description", height=200)
    
    # Optional: Add required skills
    st.subheader("Required Skills (Optional)")
    required_skills = st.text_input("Enter required skills (comma-separated)", 
                                   placeholder="Python, Machine Learning, SQL, etc.")

with col2:
    st.header("📄 Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully!")

if uploaded_files and job_description:
    st.header("🏆 Resume Ranking Results")
    
    with st.spinner("Analyzing resumes using multiple algorithms..."):
        resumes_text = []
        resume_names = []
        
        # Extract text from each uploaded PDF
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes_text.append(text)
            resume_names.append(file.name)
        
        # Calculate enhanced ATS scores
        enhanced_results = get_enhanced_ats_scores(job_description, resumes_text, resume_names)
        
        # Create DataFrame
        df_results = pd.DataFrame(enhanced_results)
        df_results = df_results.sort_values(by="composite_score", ascending=False)
        
        # Filter by minimum threshold
        df_filtered = df_results[df_results['composite_score'] >= min_score_threshold]
        
        # Display results
        if len(df_filtered) > 0:
            st.success(f"Found {len(df_filtered)} candidate(s) meeting the minimum threshold of {min_score_threshold}%")
            
            # Main results table
            display_df = df_filtered[['resume_name', 'composite_score', 'experience_years']].copy()
            display_df.columns = ['Resume', 'Overall Score', 'Years of Experience']
            display_df['Overall Score'] = display_df['Overall Score'].round(2)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Detailed scores table
            if show_detailed_scores:
                st.subheader("📊 Detailed Algorithm Scores")
                detailed_df = df_filtered[['resume_name', 'keyword_score', 'semantic_score', 
                                         'tfidf_score', 'experience_score']].copy()
                detailed_df.columns = ['Resume', 'Keyword Match', 'Semantic Similarity', 
                                     'TF-IDF Score', 'Experience Score']
                for col in ['Keyword Match', 'Semantic Similarity', 'TF-IDF Score', 'Experience Score']:
                    detailed_df[col] = detailed_df[col].round(2)
                
                st.dataframe(detailed_df, use_container_width=True)
        else:
            st.warning(f"No candidates meet the minimum threshold of {min_score_threshold}%")
            st.info("Consider lowering the threshold or reviewing the job description.")
    
    # Visualizations
    st.header("📈 Visualization")
    
    tab1, tab2, tab3 = st.tabs(["Overall Scores", "Algorithm Comparison", "Experience Distribution"])
    
    with tab1:
        # Overall scores bar chart
        fig_overall = px.bar(df_results, x='resume_name', y='composite_score',
                           title='Overall Resume Scores',
                           labels={'composite_score': 'Composite Score', 'resume_name': 'Resume'},
                           color='composite_score',
                           color_continuous_scale='RdYlGn')
        fig_overall.add_hline(y=min_score_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Minimum Threshold ({min_score_threshold}%)")
        st.plotly_chart(fig_overall, use_container_width=True)
    
    with tab2:
        # Algorithm comparison radar chart
        if len(df_results) > 0:
            # Select top 5 candidates for radar chart
            top_candidates = df_results.head(5)
            
            fig_radar = go.Figure()
            
            for idx, row in top_candidates.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['keyword_score'], row['semantic_score'], 
                       row['tfidf_score'], row['experience_score']],
                    theta=['Keyword Match', 'Semantic Similarity', 'TF-IDF Score', 'Experience Score'],
                    fill='toself',
                    name=row['resume_name'][:20] + "..." if len(row['resume_name']) > 20 else row['resume_name']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                title="Algorithm Comparison - Top 5 Candidates"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        # Experience distribution
        fig_exp = px.histogram(df_results, x='experience_years', nbins=10,
                             title='Years of Experience Distribution',
                             labels={'experience_years': 'Years of Experience', 'count': 'Number of Candidates'})
        st.plotly_chart(fig_exp, use_container_width=True)
    
    # Skills analysis (if required skills provided)
    if required_skills:
        st.header("🎯 Skills Analysis")
        skills_list = [skill.strip() for skill in required_skills.split(',')]
        
        skills_data = []
        for i, resume_text in enumerate(resumes_text):
            found_skills = extract_skills(resume_text, skills_list)
            skills_data.append({
                'Resume': resume_names[i],
                'Skills Found': ', '.join(found_skills),
                'Skills Count': len(found_skills),
                'Skills Percentage': (len(found_skills) / len(skills_list)) * 100
            })
        
        skills_df = pd.DataFrame(skills_data)
        st.dataframe(skills_df, use_container_width=True)
    
    # Personalized Suggestions Section
    st.header("💡 Personalized Suggestions for Resume Improvement")
    
    # Create tabs for each resume
    if len(df_results) > 0:
        resume_tabs = st.tabs([f"📄 {name[:15]}..." if len(name) > 15 else f"📄 {name}" 
                              for name in df_results['resume_name'].tolist()])
        
        for idx, (tab, (_, row)) in enumerate(zip(resume_tabs, df_results.iterrows())):
            with tab:
                st.subheader(f"Suggestions for {row['resume_name']}")
                st.metric("Current Score", f"{row['composite_score']:.1f}%")
                
                suggestions = row['suggestions']
                if suggestions:
                    for i, suggestion in enumerate(suggestions):
                        # Color code by priority
                        if suggestion['priority'] == 'High':
                            st.error(f"🔴 **{suggestion['category']}** (High Priority)")
                        elif suggestion['priority'] == 'Medium':
                            st.warning(f"🟡 **{suggestion['category']}** (Medium Priority)")
                        else:
                            st.info(f"🔵 **{suggestion['category']}** (Low Priority)")
                        
                        st.write(f"**Suggestion:** {suggestion['suggestion']}")
                        st.write(f"**Expected Impact:** {suggestion['impact']}")
                        st.write("---")
                else:
                    st.success("🎉 Great! This resume is well-optimized for the job description.")
                
                # Show score breakdown for context
                st.subheader("Score Breakdown")
                score_col1, score_col2 = st.columns(2)
                
                with score_col1:
                    st.metric("Keyword Match", f"{row['keyword_score']:.1f}%")
                    st.metric("Semantic Similarity", f"{row['semantic_score']:.1f}%")
                
                with score_col2:
                    st.metric("TF-IDF Score", f"{row['tfidf_score']:.1f}%")
                    st.metric("Experience Score", f"{row['experience_score']:.1f}%")
    
    # Export options
    st.header("📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export main results
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Download Main Results (CSV)",
            data=csv_data,
            file_name="resume_ranking_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export detailed results with suggestions
        if show_detailed_scores:
            # Create suggestions summary for export
            suggestions_data = []
            for _, row in df_results.iterrows():
                suggestions_text = []
                for suggestion in row['suggestions']:
                    suggestions_text.append(f"[{suggestion['priority']}] {suggestion['category']}: {suggestion['suggestion']}")
                
                suggestions_data.append({
                    'Resume': row['resume_name'],
                    'Overall Score': row['composite_score'],
                    'Top Suggestions': ' | '.join(suggestions_text[:3])  # Top 3 suggestions
                })
            
            suggestions_df = pd.DataFrame(suggestions_data)
            suggestions_csv = suggestions_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💡 Download Suggestions Report (CSV)",
                data=suggestions_csv,
                file_name="resume_suggestions_report.csv",
                mime="text/csv"
            )
    
    # Algorithm explanation
    with st.expander("🔍 Algorithm Explanation"):
        st.markdown("""
        ### Scoring Methodology
        
        The composite score is calculated using multiple algorithms with weighted contributions:
        
        1. **Keyword Matching (30%)**: Extracts important keywords from job description using TF-IDF and measures their presence in resumes
        2. **Semantic Similarity (35%)**: Uses sentence transformers to measure semantic similarity between job description and resume
        3. **TF-IDF Similarity (20%)**: Traditional text similarity using Term Frequency-Inverse Document Frequency
        4. **Experience Score (15%)**: Extracts years of experience mentioned in resume and converts to score
        
        ### Key Improvements:
        - **Multi-algorithm approach** for more accurate scoring
        - **Text preprocessing** for better keyword extraction
        - **Experience extraction** using regex patterns
        - **Configurable thresholds** for filtering candidates
        - **Detailed visualizations** for better insights
        - **Skills analysis** for specific requirements
        - **Personalized suggestions** to improve ATS scores
        - **Priority-based recommendations** for maximum impact
        """)

else:
    st.info("👆 Please upload resume files and enter a job description to begin the analysis.")
    
    # Example section
    with st.expander("📖 Example Usage"):
        st.markdown("""
        ### How to Use:
        1. **Enter Job Description**: Copy and paste the complete job description
        2. **Add Required Skills** (Optional): List specific skills you're looking for
        3. **Upload Resumes**: Select multiple PDF files
        4. **Adjust Settings**: Use the sidebar to configure scoring thresholds
        5. **Review Results**: Analyze the ranking and detailed scores
        6. **Export Data**: Download results for further analysis
        
        ### Tips for Better Results:
        - Use comprehensive job descriptions with specific requirements
        - Include both hard and soft skills in the job description
        - Upload resumes in consistent PDF format
        - Adjust the minimum score threshold based on your needs
        """)
