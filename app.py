from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load trained Spam model
with open("spam_detection.pkl", "rb") as f:
    model = pickle.load(f)

# ⚠️ Must match your training values
dic_size = 5500     # from your code
maxlen = 80       # or use the same `self.a` you trained with

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = ''.join([i for i in text if i not in string.punctuation])
    text = ' '.join([lemmatizer.lemmatize(i) for i in text.split() if i not in stop_words])
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        mail = request.form["review"]
        cleaned = clean_text(mail)

        v = [one_hot(cleaned, dic_size)]
        p = pad_sequences(v, maxlen=maxlen, padding='post')

        pred = model.predict(p)

        if pred[0][0] >= 0.5:
            prediction = "Spam Mail"
        else:
            prediction = "Ham Mail"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)