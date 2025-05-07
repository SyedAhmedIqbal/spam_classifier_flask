import pandas as pd
import nltk
import string, joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"D:\superior\semester_4th\PAI theory\assignment 3\spam.csv", encoding="ISO-8859-1")
df = data[['v1', 'v2']].copy()
df.columns = ['label', 'message']

def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

df['cleaned_message'] = df['message'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#def save_model_and_vectorizer(model, vectorizer, model_path="spam_classifier_model.pkl", vectorizer_path="vectorizer.pkl"):
 #   joblib.dump(model, model_path)
  #  joblib.dump(vectorizer, vectorizer_path)
   # print(f"Model saved to {model_path}")
   # print(f"Vectorizer saved to {vectorizer_path}")

import pickle

def save_model_and_vectorizer(model, vectorizer, model_path="spam_classifier_model.pkl", vectorizer_path="vectorizer.pkl"):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
save_model_and_vectorizer(model,vectorizer)