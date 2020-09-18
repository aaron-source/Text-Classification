from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time
from Bio import Medline, Entrez
import numpy as np
from Bio import Medline, Entrez
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
Data =  pd.read_csv('diseases-train.csv')
Entrez.email = "aaronzellner1234@gmail.com"
def clean_data(Data_file):
  edit_String = ""
  data_list = []
  for index, data in Data_file.iterrows():
      doc_1 = data[0]
      doc_2 = Entrez.efetch(db="pubmed", id=doc_1, rettype="medline", retmode="text")
      journals = Medline.read(doc_2)
      new_data = pre_processor(journals)
      data_list.append(new_data)
  return data_list


def pre_processor(info):
  Analyze = LancasterStemmer()
  edit_String = ""
  new_string = ""
  word_list = []
  for key, values in info.items():
    if key == 'AB':
      values = ''.join(str(value) for value in values)
      new_values = re.sub("[^A-Za-z0-9]",  " ", values)
      new_words = word_tokenize(new_values)
      for word in new_words: 
            word = Analyze.stem(word)
            word_list.append(word)
      adjusted = ' '.join(w for w in word_list)
      new_string = new_string + adjusted
  return new_string 


train_1, train_2 = train_test_split(Data, test_size=0.2, random_state=42)
Data_ix = clean_data(train_1)
Data_x = clean_data(train_2)
#declaring my prediction model and feature selection model
evaluate = MultinomialNB()

#Testing my first training set of data on the MB model
fit_Dataix = count.fit_transform(Data_ix)
evaluate.fit(fit_Dataix, train_1.category)
evaluate.predict(fit_Dataix)
count = CountVectorizer(stop_words="english")

#Testing my seconding training set of data on the MB model
fit_Data_x = count.fit_transform(Data_x)
evaluate.fit(fit_Data_x, train_2.category)
evaluate.predict(fit_Data_x)

#making a smaller training set so it fits the size of the test set
test_1, test_2 = train_test_split(train_2, test_size=0.55, random_state=42)

#Testing a file with no labels(categories)
new_file = pd.read_csv("Test_set.csv")
Data_test = clean_data(new_file)
test_fit = count.fit_transform(Data_test)
evaluate.fit(test_fit, test_2.category)
evaluate.predict(test_fit)
