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

def matcher(job_description_keywords,resume_keywords):
    missing_keywords = [keyword for keyword in job_description_keywords if keyword not in resume_keywords]
    vectorizer = TfidfVectorizer()
    job_description_vector = vectorizer.fit_transform([job_description_text])
    resume_vector = vectorizer.transform([resume_text])
    similarity_score = cosine_similarity(job_description_vector, resume_vector)[0][0]
    fitment_rating = similarity_score * 100
	
    
    return fitment_rating, missing_keywords

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
    
model_id = 'google/flan-t5-small'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map = 'auto')
pipeline = pipeline(
    "text2text-generation",
    model = model,
    tokenizer= tokenizer,
    max_length = 1028
)
local_llm = HuggingFacePipeline(pipeline=pipeline)


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

job_description_keywords = local_llm(f"You are a ATS service for technical jobs. From the given job description find all the keywords that are in any way related to technology or programming languages.Only output words. Job Description: {job_description_text}.")
resume_keywords = local_llm(f"You are a ATS service for technical jobs. From the given resume find all the keywords that are in any way related to technology or programming languages.Only output words. Job Description: {resume_text}.")
print(job_description_keywords)
print(resume_keywords)
matcher(job_description_keywords, resume_keywords)