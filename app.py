# from fastapi import FastAPI, Request

# app = FastAPI()

from flask import Flask, request


app = Flask(__name__, static_folder='public')

#mendapatkan data artikel dari api
import requests
import json
import pandas as pd

response = requests.get('https://nuha.my.id/api/video')
data = response.text
parse_json = json.loads(data)
data_dict = parse_json['data']
df_orgdata = pd.json_normalize(data_dict)

# membuat dataframe & preprocessing data
import string

content_list = []

for content in list(data_dict):
    # case folding
    title = content["title"].lower().replace("/n"," ").replace("\n"," ").replace("\r"," ")
    category = content["category"].lower().replace("/n"," ").replace("\n"," ").replace("\r"," ")
    description = content["description"].lower().replace("/n"," ").replace("\n"," ").replace("\r"," ")

    # cleaning tanda baca 
    for ch in string.punctuation:
        title = title.replace(ch, "").replace("”","").replace("“","").replace("’","")
        category = category.replace(ch, "").replace("”","").replace("“","").replace("’","")
        description = description.replace(ch, "").replace("”","").replace("“","").replace("’","")
    # print(title)

    values = {"title":title, "category":category, "description":description}

    content_list.append(pd.DataFrame.from_records([values]))

df = pd.concat(content_list, ignore_index=True)
df["desc"] = df["title"] + " " + df["category"] + " " + df["description"]
text = df["desc"]

# stopwords removal
import nltk
nltk.download('stopwords')
nltk.download('punkt') 
from nltk.corpus import stopwords

indo = stopwords.words('indonesian')
eng = stopwords.words('english')
stopwords_list = indo
stopwords_list.extend(eng)

def stopwords_removal(text):
    text = " ".join(word for word in text.split() if word not in stopwords_list)
    return text

text = text.apply(stopwords_removal)

# tokenize
from nltk.tokenize import word_tokenize 
text = text.apply(word_tokenize)

#stemming
import re
import swifter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import ast
factory = StemmerFactory() 
stemmer = factory.create_stemmer() 

# stemmed term
def stemmed_wrapper(term): 
  return stemmer.stem(term) 

term_dict = {} 
for document in text: 
  for term in document: 
    if term not in term_dict: 
      term_dict[term] = ' ' 

for term in term_dict: 
  term_dict[term] = stemmed_wrapper(term) 

# apply stemmed term to dataframe 
def get_stemmed_term(document): 
  terms = [term_dict[term] for term in document] 
  return terms
text = text.apply(get_stemmed_term)

#join list of terms
def join_text_list(texts):
  texts : ast.literal.eval(texts)
  return " ".join([text for text in texts])

text = text.swifter.apply(join_text_list)

# TF-IDF Vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

documents = text

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Menghitung matrix cosine similarity 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 

# @app.get('/recommend/video')
@app.route('/video')
def hello():
    return 'nuha video recommender sudah siap!'

# @app.get('/recommend/video')
@app.route('/recommend/video')
def get_recommendation() :
    id = int(request.args.get('id'))
    id -= 1
    sim_score = enumerate(cosine_sim[id])
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:4]
    sim_index = [i[0] for i in sim_score]
    return {"data" : df_orgdata.iloc[sim_index].to_dict(orient='records')}

#uvicorn app:app --reload

#pip install -r requirements.txt
#uvicorn app:app --host 0.0.0.0 --port $PORT
