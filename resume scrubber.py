import pandas as pd
import re

bias=pd.read_excel("ProcessedBias.xlsx")
bias.fillna("",inplace=True)
resume = """Resume:Barry Allen
Front-End Developer Google HQ, Mountain View, CA, USA
Mobile-(123) 456-7890
envelpeallthe.stars@google.com
linkedin-inbwayne
github-wayne
Objective
being a tacos
Seeking a challenging role as a Front-End Developer where I can leverage my knowledge of UI/UX
design and modern web technologies to create intuitive and engaging user interfaces.
Education
2018 – 2022 B.Tech, Computer Science and Engineering ,Indian Institute of Technology
Delhi , New Delhi, India
Overall GPA: 9.5/10
Skills
JavaScript (ES6+), TypeScript, HTML5, CSS3, Python, React, Redux, Angular,
Vue.js, Node.js, Express.js, D3.js, Git, Docker, Webpack, Babel, Google Cloud
Platform, Firebase, RESTful APIs, GraphQL, Agile Development, Test-Driven
Development, Responsive Design, UI/UX
Experience
June 2022 –
PresentSoftware Engineer (Front-End) ,Google , Mountain View, CA, USA
Developing intuitive and engaging user interfaces using React and Redux. Working closely
with UX designers to implement responsive and accessible web design. Participating in
agile development processes, including sprint planning and code reviews. Collaborating with
back-end developers to integrate RESTful APIs and ensure seamless data flow.
Projects
2022 Personal Expense Tracker
Developed a personal expense tracker application using React, Redux, and Firebase. Imple-
mented user authentication using Firebase Auth and data storage using Firestore. Utilized
D3.js for data visualization to provide users with insights into their spending patterns."""
lines=[i.lower() for i in resume.split("\n") if len(i)>0]
k=[re.sub(r"\n"," ",re.sub(r"[^A-Za-z0-9\s.@/\\,]","",m)).strip() for m in lines]
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
print("\n".join(refined))
