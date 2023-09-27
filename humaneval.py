import pandas as pd
from sklearn.metrics import mean_absolute_error
import resumematcher
def process_excel_sheet(excel_file):
    dataset = pd.read_excel(excel_file, engine="openpyxl")
    pred_fitment = []
    actual_fitment = []

    for index, row in dataset.iterrows():
        job_description = row["Job Description"]
        resume = row["Resume"]
        actual_fitment_rating = row["Fitment Rating"]

        fitment_score = resumematcher.matcher(resume, job_description)
        pred_fitment.append(fitment_score)
        actual_fitment.append(actual_fitment_rating)

    mae = mean_absolute_error(actual_fitment, pred_fitment)
    return mae

pred_fitment, actual_fitment, mae = process_excel_sheet("your_dataset.xlsx")
print(f"Mean Absolute Error (MAE): {mae}")
