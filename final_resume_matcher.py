import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import spacy
import transformers 
import PyPDF2
from docx import Document
import os 
from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import fitz
import requests
import mistune
import pandas as pd
from sklearn.metrics import mean_absolute_error
from bs4 import BeautifulSoup
import re
import spacy
from collections import Counter
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from huggingface_hub import hf_hub_download
HUGGING_FACE_API_KEY = ""
model_id = "lmsys/fastchat-t5-3b-v1.0"
#filenames = [
    #"pytorch_model.bin","added_tokens.json","config.json","generation_config.json",
            #"special_tokens_map.json","spiece.model", "tokenizer_config.json"]
#for filename in filenames:
    #downloaded_model_path = hf_hub_download(repo_id=model_id,filename=filename,token=HUGGING_FACE_API_KEY)
    #print(downloaded_model_path)


def convert_github_url_to_raw_url(url):
    """Converts a GitHub URL to a raw URL.

    Args:
        url: A GitHub URL.

    Returns:
        A raw URL.
    """

    # Remove the "blob/" part of the URL.
    url = re.sub(r'/blob/', '/', url)

    # Add the "raw/" part of the URL.
    url = url.replace('github.com', 'raw.githubusercontent.com')

    return url


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy = False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length = 400)


def match_resume_to_job_description(job_description, resume):
    
    resume_keywords = extract_keywords(resume_text)
    job_description_keywords = extract_keywords(job_description_text)
    missing_keywords = [keyword for keyword in job_description_keywords if keyword not in resume_keywords]
    
    vectorizer = TfidfVectorizer()
    job_description_vector = vectorizer.fit_transform([job_description])
    resume_vector = vectorizer.transform([resume])
    similarity_score = cosine_similarity(job_description_vector, resume_vector)[0][0]
    fitment_rating = similarity_score * 100
	
    
    return fitment_rating, missing_keywords




def extract_keywords(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return keywords

# Example usage:
text = "This is an example sentence for extracting keywords 123 using NLTK."
keywords = extract_keywords(text)
print(keywords)


def matcher(job_description_keywords,resume_keywords):
    missing_keywords = [keyword for keyword in job_description_keywords if keyword not in resume_keywords]
    vectorizer = TfidfVectorizer()
    job_description_vector = vectorizer.fit_transform([job_description_text])
    resume_vector = vectorizer.transform([resume_text])
    similarity_score = cosine_similarity(job_description_vector, resume_vector)[0][0]
    fitment_rating = similarity_score * 100
	
    #return_text(missing_keywords)
    return fitment_rating, missing_keywords


def confidenceScore(resume_keywords, job_description_keywords):
    extra_keywords = []
    union = []
    skills = ['Communicate', 'Team', 'Solve', 'Lead', 'Manage', 'Adapt', 'Interpersonal', 'Create', 'Organize', 'Work', 'Stress', 'Tech', 'Serve', 'Finance', 'Culture', 'Conflict', 'Network', 'Ethical', 'Emotion', 'Project', 'Critical', 'Decide', 'Initiate', 'Account', 'Detail', 'Plan', 'Resilient', 'Goal', 'Speak', 'Negotiate', 'Research', 'Delegate', 'Risk', 'Patient', 'Persuade', 'Analyze', 'Coach', 'Present', 'Build', 'Analyse', 'Time', 'Motivate', 'Innovate', 'Technique', 'Client', 'Sale', 'Coordinate', 'Identify', 'Crisis']
    for keyword in resume_keywords:
        if keyword not in job_description_keywords and keyword not in skills:
            extra_keywords.append(keyword)

    for keyword in resume_keywords:
        if keyword not in extra_keywords:
            union.append(keyword)

    confidence_score = (len(union)/len(extra_keywords))*100
    print(f"confidence score = {confidence_score}")
    return confidence_score

def process_excel_sheet(excel_file):
    dataset = pd.read_excel(excel_file, engine="openpyxl")
    pred_fitment = []
    actual_fitment = []

    for index, row in dataset.iterrows():
        job_description = row["Job Description"]
        resume = row["Resume"]
        actual_fitment_rating = row["Fitment Rating"]

        fitment_score = matcher(job_description, resume)
        pred_fitment.append(fitment_score)
        actual_fitment.append(actual_fitment_rating)
        
    mae = mean_absolute_error(actual_fitment, pred_fitment)
    return mae


def read_data_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def read_data_from_docx(docx_path):
    document = Document(docx_path)
    paragraphs = document.paragraphs
    text = ''
    for paragraph in paragraphs:
        text += paragraph.text
        text += '\n\n'
    return text

def identify_file_type(file_path):
    if file_path.endswith('.pdf'):
        return 'PDF'
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        return 'Word Document'
    else:
        return 'Unknown'
    

resume_file_path = 'barry_allen_fe.pdf'
jd_file_path = 'sample_job_description.pdf'

file_type = identify_file_type(resume_file_path)
if file_type == 'Word Document':
    resume_text = read_data_from_docx(resume_file_path)
    
elif file_type == 'PDF':
    resume_text = read_data_from_pdf(resume_file_path)

file_type = identify_file_type(jd_file_path)
if file_type == 'Word Document':
    job_description_text = read_data_from_docx(jd_file_path)
    
elif file_type == 'PDF':
    job_description_text = read_data_from_pdf(jd_file_path)


#web_text = extract_info_from_pdf(resume_file_path)
job_description_keywords = pipeline(f"Extract all technical words from this text : {job_description_text}.")
resume_keywords = pipeline(f"Extract all technical words from this text : {resume_text}.")
#web_keywords = pipeline(f"Extract all technical words from this text : {web_text}.")
#esume_keywords = resume_keywords + web_keywords

#print(job_description_keywords)
#print(resume_keywords)
match_resume_to_job_description(job_description_keywords, resume_keywords)
mae_result = process_excel_sheet("resume.xlsx")
print("accuracy : ", mae_result)
confidenceScore(job_description_keywords,resume_keywords)

