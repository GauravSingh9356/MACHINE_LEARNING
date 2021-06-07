# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:05:22 2021

@author: gs9356
"""

#imports

import numpy as np
import pandas as pd
import re
import nltk


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


dataset = pd.read_csv("data.csv", sep="\t", names=["label", "message"])

##data preprocessing
corpus=[]
for i in range(len(dataset)):
    review = dataset["message"][i]
    review = re.sub("[^a-zA-Z]"," ", review)
    review = review.lower()
    review = review.split(" ")
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    


##bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


##testing 
text="Ok lar... Joking wif u oni..."
text = text.lower()
text = text.split(" ")
text= [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
text = " ".join(text)
print(text)



text = cv.transform([text]).toarray()

print(nb.classes_)


text.shape


ans = nb.predict(text) 





