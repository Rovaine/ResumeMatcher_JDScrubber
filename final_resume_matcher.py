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
from huggingface_hub import hf_hub_download


HUGGING_FACE_API_KEY = "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"
model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
    "pytorch_model.bin","added_tokens.json","config.json","generation_config.json",
             "special_tokens_map.json","spiece.model", "tokenizer_config.json#"]
for filename in filenames:
    downloaded_model_path = hf_hub_download(repo_id=model_id,filename=filename,token=HUGGING_FACE_API_KEY)
    print(downloaded_model_path)


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

def extract_info_from_pdf(pdf_path):
    """Extracts information from GitHub URLs in a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of extracted information from GitHub URLs.
    """
    doc = fitz.open(pdf_path)
    lnks = []
    
    # Get the links on all pages
    for i in range(doc.page_count):
        page = doc.load_page(i)
        link = page.get_links()
        
        # Print the actual links stored under the key "uri"
        for obj in link:
            if 'github.com' in obj["uri"]:
                lnks.append(obj["uri"])
    
    extracted_info = []
    
    for lnk in lnks:
        github_url = lnk + "/blob/main/README.md"
        url = convert_github_url_to_raw_url(github_url)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
        }

        response = requests.get(github_url, headers=headers)

        try:
            response.raise_for_status()  # Raise an exception for HTTP errors

            markdown_content = response.text

            # Parse Markdown content using mistune with HTML rendering
            markdown_parser = mistune.create_markdown(renderer=mistune.HTMLRenderer())
            html_content = markdown_parser(markdown_content)

            # Now you can work with the parsed HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find the <p> tag with dir='auto' and extract its text
            p_tags = soup.select('p', {'dir': 'auto'})
            info1 = []
            for p_tag in p_tags:
                info1.append(p_tag.get_text())
            pattern = r'<p dir="auto">(.*?)<\/p>'
            matches = re.findall(pattern, info1[0], re.DOTALL)
            info = []

            # Extracted text
            for match in matches:
                info.append(match.strip())
            clean_info = [re.sub(r'<[^>]*>', '', item).strip() for item in info if item.strip()]

            # Define a regular expression to match emojis and symbols
            emoji_pattern = re.compile("["
                                        u"\U0001F600-\U0001F64F"  # Emoticons
                                        # Add more Unicode ranges for symbols as needed
                                        "]+", flags=re.UNICODE)

            # Remove emojis, symbols, and empty strings
            info = [re.sub(emoji_pattern, '', item).strip() for item in clean_info if item.strip()]
            res = ' '.join(info)
            # print(res)
            
            # function to extract keywords
            # spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            def get_hotwords(text):
                result = []
                pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
                doc = nlp(text.lower()) 
                for token in doc:
                    if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
                        continue
                    if(token.pos_ in pos_tag):
                        result.append(token.text)
                return result
            new_text = res
            output = set(get_hotwords(new_text))
            most_common_list = Counter(output).most_common(10)
            for item in most_common_list:
                extracted_info.append(item[0])
            # extracted_info.append(info)
        except requests.exceptions.RequestException as e:
            print(f"Error while processing URL {github_url}: {str(e)}")
            continue  # Continue with the next URL
    
    return extracted_info


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy = False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length = 400)



def process_excel_sheet(excel_file):
    dataset = pd.read_excel(excel_file, engine="openpyxl")
    pred_fitment = []
    actual_fitment = []

    for index, row in dataset.iterrows():
        job_description = row["Job Description"]
        resume = row["Resume"]
        actual_fitment_rating = row["Fitment Rating"]

        fitment_score = matcher(resume, job_description)
        pred_fitment.append(fitment_score)
        actual_fitment.append(actual_fitment_rating)

    mae = mean_absolute_error(actual_fitment, pred_fitment)
    return mae

pred_fitment, actual_fitment, mae = process_excel_sheet("your_dataset.xlsx")
print(f"Mean Absolute Error (MAE): {mae}")


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


web_text = extract_info_from_pdf(resume_file_path)
job_description_keywords = pipeline(f"Extract all technical words from this text : {job_description_text}.")
resume_keywords = pipeline(f"Extract all technical words from this text : {resume_text}.")
web_keywords = pipeline(f"Extract all technical words from this text : {web_text}.")
resume_keywords = resume_keywords + web_keywords

#print(job_description_keywords)
#print(resume_keywords)
matcher(job_description_keywords, resume_keywords)
print("accuracy : ", process_excel_sheet("excelsheet.xlsx"))
confidenceScore(job_description_keywords,resume_keywords)



