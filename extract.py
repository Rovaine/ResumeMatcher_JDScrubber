# This file helps to extract the content from the Readme.md file of the GitHub project repos mentioned in the resume 

# The import name for this library is fitz
import fitz
import requests
import mistune
from bs4 import BeautifulSoup
import re

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

# Create a document object
doc = fitz.open('E:\Web Scrapping\Biswatosh_Mazumder_Resume.pdf')  # or fitz.Document(filename)
lnks=[]
# get the links on all pages
for i in range(doc.page_count):
  page = doc.load_page(i)
  link = page.get_links()
  # print the actual links stored under the key "uri"
  for obj in link:
    # print(obj["uri"])
    if 'github.com' in obj["uri"]:
      lnks.append(obj["uri"])
# print(lnks)

ll=["https://github.com/rodrigomasiniai/ResumeScreeningApp","https://github.com/Biswatosh01/Construction-Site-Safety-","https://github.com/srbhr/Resume-Matcher"]
for lnk in ll:
    github_url = lnk + "/blob/main/README.md"
    url = convert_github_url_to_raw_url(github_url)
    print("\n\nGitHub repo Content:\n\n")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        markdown_content = response.text

        # Parse Markdown content using mistune with HTML rendering
        markdown_parser = mistune.create_markdown(renderer=mistune.HTMLRenderer())
        html_content = markdown_parser(markdown_content)

        # Now you can work with the parsed HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the <p> tag with dir='auto' and extract its text
        p_tags = soup.select('p', {'dir': 'auto'})
        info=[]
        for p_tag in p_tags:
            info.append(p_tag.get_text())
        clean_info = [re.sub(r'<[^>]*>', '', item).strip() for item in info if item.strip()]
        # Define a regular expression to match emojis and symbols
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # Emoticons
                                u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                                u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                                u"\U0001F700-\U0001F77F"  # Alphabetic Presentation Forms
                                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                u"\U0001F004-\U0001F0CF"  # Mahjong Tiles
                                u"\U0001F170-\U0001F251"  # Enclosed Ideographic Supplement
                                "]+", flags=re.UNICODE)

        # Remove emojis, symbols, and empty strings
        info = [re.sub(emoji_pattern, '', item).strip() for item in clean_info if item.strip()]

        print(info)
    else:
        print("Failed to retrieve the README file. Status code:", response.status_code)

