# This file helps to extract the content from the Readme.md file of the GitHub project repos mentioned in the resume 


# Example usage:
# from extract import extract_info_from_pdf
# pdf_path = 'E:\\Web Scrapping\\Biswatosh_Mazumder_Resume.pdf'
# info_list = extract_info_from_pdf(pdf_path)
# for info in info_list:
#     print(info)


import fitz
import requests
import mistune
from bs4 import BeautifulSoup
import re
import spacy
from collections import Counter
from string import punctuation

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

