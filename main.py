import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords

df = pd.read_csv('C:/Users/rocks/OneDrive/Desktop/Projects/SPAM-AI/spam.csv', encoding='latin-1')
df.head()

# get necessary columns for processing
df = df[['v2', 'v1']]
# df.rename(columns={'v2': 'messages', 'v1': 'label'}, inplace=True)
df = df.rename(columns={'v2': 'messages', 'v1': 'label'})
df.head()

# check for null values
df.isnull().sum()

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

# clean the messages
df['clean_text'] = df['messages'].apply(clean_text)
df.head()

# Input Split
X = df['clean_text']
y = df['label']

# Model Training
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def classify(model, X, y):
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    # model training
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', model)])
    pipeline_model.fit(x_train, y_train)
    
    print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
    
#     cv_score = cross_val_score(model, X, y, cv=5)
#     print("CV Score:", np.mean(cv_score)*100)
    y_pred = pipeline_model.predict(x_test)
    print(classification_report(y_test, y_pred))

# LogisticRegression Model
print("LogisticRegression Model")
print("------------------------")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)
print("======================================================")

# MultinomialNB Model
print("MultinomialNB Model")
print("-------------------")
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
classify(model, X, y)
print("======================================================")

# SVC Model
print("SVC Model")
print("---------")
from sklearn.svm import SVC
model = SVC(C=3)
classify(model, X, y)
print("======================================================")

# RandomForestClassifier Model
print("RandomForestClassifier Model")
print("----------------------------")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)
print("======================================================")