#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset 
data = pd.read_csv('C:\\Users\\madan\\OneDrive\\Desktop\\spam.csv', encoding='latin1')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf, X_test_tfidf = tfidf_vectorizer.fit_transform(X_train), tfidf_vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB().fit(X_train_tfidf, y_train)

# Make predictions on the test set
predictions = nb_classifier.predict(X_test_tfidf)

# Evaluate the model's accuracy and display a classification report
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report(y_test, predictions))


# In[ ]:




