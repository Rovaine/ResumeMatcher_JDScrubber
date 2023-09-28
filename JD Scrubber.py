#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re

bias=pd.read_excel("ProcessedBias.xlsx")
bias.fillna("",inplace=True)
jobdescription = """Job Description:
Front-End Developer
Indian Black
black
Black
Male
His
Company: Google
Location: Mountain View, CA, USA

**Objective:**
We are looking for a highly motivated Front-End Developer to join our team at Google HQ in Mountain View, California. In this role, you will have the opportunity to leverage your expertise in UI/UX design and modern web technologies to create intuitive and engaging user interfaces that will impact millions of users worldwide.

**Key Responsibilities:**
- Develop intuitive and engaging user interfaces using React and Redux.
- Work closely with UX designers to implement responsive and accessible web design.
- Participate in agile development processes, including sprint planning and code reviews.
- Collaborate with back-end developers to integrate RESTful APIs and ensure seamless data flow.
- Stay up-to-date with the latest front-end development trends and best practices.

**Qualifications:**
- Bachelor's degree in Computer Science or related field.
- Proficiency in JavaScript (ES6+), TypeScript, HTML5, and CSS3.
- Experience with front-end frameworks and libraries such as React, Angular, or Vue.js.
- Familiarity with Node.js, Express.js, and modern build tools like Webpack and Babel.
- Knowledge of version control systems, particularly Git.
- Experience with cloud platforms like Google Cloud Platform and Firebase.
- Strong problem-solving skills and attention to detail.
- Excellent communication and collaboration abilities.
- A passion for creating exceptional user experiences.

**Experience:**
- Previous experience as a Front-End Developer or similar role is a plus.
- Experience with RESTful APIs and GraphQL is advantageous.
- Familiarity with Agile Development and Test-Driven Development methodologies is a bonus.

**Projects:**
- 2022 Personal Expense Tracker
  - Developed a personal expense tracker application using React, Redux, and Firebase.
  - Implemented user authentication using Firebase Auth and data storage using Firestore.
  - Utilized D3.js for data visualization to provide users with insights into their spending pattern.

If you are a talented Front-End Developer with a passion for creating exceptional user interfaces and want to be part of a dynamic team at Google, we encourage you to apply and help shape the future of web development. Join us in our mission to make the web more intuitive and engaging for users around the world."""
lines=[i.lower() for i in jobdescription.split("\n") if len(i)>0]
k=[re.sub(r"\n"," ",re.sub(r"[^A-Za-z0-9\s]","",m)).strip() for m in lines]
refined=[]
removedLines=[]
hitCount=[]
for idx in range(len(k)):
    line=k[idx]
    count=0
    priority=True
    for column in bias.columns.tolist():
        for word in bias[column]:
            if word in line.split() and len(word)>0:
                count+=1
                if column in ["Religion", "Sexuality", "Marital Status"]:
                    priority=False
    if count!=0:
        hitCount.append((count,line))
    if priority==False and count==0 or priority==True and count<2:
        refined.append(line)
    else:
        removedLines.append((count,line))

print("Refined:")
print(refined)
print("Removed Lines:")
print(removedLines)
print("Allowed Hits:")
print(hitCount)