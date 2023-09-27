# from django.shortcuts import render
# from django.http import HttpResponse
# import pdfplumber
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def home(request):
#     # if request.method == 'POST':
#     #     uploaded_jd = request.FILES['uploadedJD'].read().decode('utf-8')
#     #     uploaded_resume = request.FILES['uploadedResume'].read().decode('utf-8')
#     #     return render(request,'result.html')

#         # get_result(request,uploaded_jd, uploaded_resume)
#         # match=round(match,2)
#         # data={
#         #     'result':match,
#         #     'jd':uploaded_jd,
#         #     'rs':uploaded_resume
#         # }
#         # return render(request,'result.html',data)

#     return render(request, 'index.html')

# def get_result(request):
# # # def get_result(request):
# #     content = [jd_txt, resume_txt]

# #     cv = CountVectorizer()
# #     matrix = cv.fit_transform(content)

# #     similarity_matrix = cosine_similarity(matrix)

# #     match = similarity_matrix[0][1] * 100
# #     return render(request, 'result.html')
#     if request.method == 'POST':
#             uploaded_jd = request.FILES['uploadedJD'].read().decode('utf-8')
#             uploaded_resume = request.FILES['uploadedResume'].read().decode('utf-8')
#             return render(request,'result.html')
import os
import PyPDF2
from django.shortcuts import render, redirect
from django.http import JsonResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    text = ''
    try:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page_num).extractText()
    except Exception as e:
        # Handle any exceptions here
        pass
    return text

# View to upload PDFs
def upload_pdf(request):
    if request.method == 'POST':
        pdf1 = request.FILES['pdf1']
        pdf2 = request.FILES['pdf2']
        pdf1_text = extract_text_from_pdf(pdf1)
        pdf2_text = extract_text_from_pdf(pdf2)

        # Save the extracted text to temporary files
        with open('temp1.txt', 'w', encoding='utf-8') as temp1_file:
            temp1_file.write(pdf1_text)
        with open('temp2.txt', 'w', encoding='utf-8') as temp2_file:
            temp2_file.write(pdf2_text)

        return redirect('compare_pdfs')
    
    return render(request, 'index.html')

# View to compare PDFs and calculate cosine similarity
def compare_pdfs(request):
    with open('temp1.txt', 'r', encoding='utf-8') as temp1_file:
        text1 = temp1_file.read()
    with open('temp2.txt', 'r', encoding='utf-8') as temp2_file:
        text2 = temp2_file.read()

    # Calculate cosine similarity between text1 and text2
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)

    similarity_percentage = cosine_sim[0][1] * 100

    return render(request, 'results.html', {'similarity_percentage': similarity_percentage},{'jd':text1},{'rs':text2})

# View to display results
def results(request):
    similarity_percentage = request.GET.get('similarity_percentage', None)
    return render(request, 'result.html', {'similarity_percentage': similarity_percentage})
