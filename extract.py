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

ll=["https://github.com/rodrigomasiniai/ResumeScreeningApp","https://github.com/Biswatosh01/Construction-Site-Safety-"]
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
        print(info)
    else:
        print("Failed to retrieve the README file. Status code:", response.status_code)

