import pandas as pd
from sklearn.metrics import mean_absolute_error
import resumematcher
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

# Download the 'stopwords' and 'punkt' resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Rest of your code...



# Load the spaCy model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Provide the path to your Excel file
excel_file_path = 'resume.xlsx'

# Read the Excel sheet into a pandas DataFrame
df = pd.read_excel(excel_file_path)

def process_excel_sheet(excel_file):
    dataset = pd.read_excel(excel_file, engine="openpyxl")
    pred_fitment = []
    actual_fitment = []

    for index, row in dataset.iterrows():
        job_description = str(row["Job-Description"])  # Convert to string
        resume = str(row["Resume"])  # Convert to string
        actual_fitment_rating = row["Fitment-Rating"]

        fitment_score = resumematcher.match_resume_to_job_description(resume, job_description)
        pred_fitment.append(fitment_score)
        actual_fitment.append(actual_fitment_rating)
    # Convert lists to NumPy arrays
    actual_fitment = np.array(actual_fitment)
    # Extract the first element (fitment score) from each tuple or list
    pred_fitment_flat = [item[0] for item in pred_fitment]

    # Convert the flattened list to a NumPy array
    pred_fitment_array = np.array(pred_fitment_flat)
    mae = mean_absolute_error(actual_fitment, pred_fitment_array)
    return mae

if __name__ == "__main__":
    mae = process_excel_sheet("resume.xlsx")
    print(f"Mean Absolute Error (MAE): {mae}")

def tokenize_excel_columns(file_path, sheet_name, column1_name, column2_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    combined_column = df[column1_name] + " " + df[column2_name]
    keywords = []
    stop_words = set(stopwords.words("english"))
    for text in combined_column:
        text = str(text)  # Convert to string
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
        keywords.extend(tokens)

    return keywords

# Define a function to extract keywords from text
def text_to_keywords(text):
    text = str(text)  # Convert to string
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "ADJ")]
    return keywords

# Iterate through text columns and convert text to keywords
# Replace with your column names
text_columns = ['Job-Description', 'Resume', 'Fitment-Rating']
for column_name in text_columns:
    df[column_name + '_Keywords'] = df[column_name].apply(text_to_keywords)

# Now, your DataFrame contains new columns with keywords extracted from text columns
# You can access these keyword columns like df['Column1_Keywords']

# Save the DataFrame with keyword columns to a new Excel file if needed
df.to_excel('output_with_keywords.xlsx', index=False)
excel_file = "resume.xlsx"
sheet_name = "Sheet1"
column1_name = "Job-Description"
column2_name = "Resume"
keywords = tokenize_excel_columns(excel_file, sheet_name, column1_name, column2_name)
print(keywords)
