from flask import Flask, request, render_template
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pickle

# Ensure NLTK data is available (you can pre-download these on your local machine and upload them if needed)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def preprocess_text(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join(char for char in text if char not in string.punctuation)
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

def load_model_and_vectorizer():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'spam_classifier_model.pkl')
        vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model or vectorizer: {e}")
model, vectorizer = load_model_and_vectorizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            try:
                processed = preprocess_text(message)
                transformed = vectorizer.transform([processed])
                prediction = model.predict(transformed)
                result = "Spam" if prediction[0] == 1 else "Ham"
            except Exception as e:
                result = f"Error during prediction: {e}"
        else:
            result = "Please enter a message."
    return render_template("index.html", result=result, message=message)

if __name__ == '__main__':
    # For local testing only – PythonAnywhere will use your WSGI config
    app.run(host='0.0.0.0', port=5000, debug=True)
