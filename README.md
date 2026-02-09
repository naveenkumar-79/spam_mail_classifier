# spam_mail_classifier

📌 **Project Overview**

This project is an Email Spam Classification System built using Natural Language Processing (NLP) and Machine Learning / Deep Learning techniques. The model analyzes the content of an email and classifies it as either Spam or Ham (Not Spam).

The system demonstrates a complete NLP pipeline—from raw text preprocessing to model inference—using a Bidiretional LSTM classification model and saved the model using pickle.

🚀 **Features**

Classifies emails as Spam or Ham

Uses real-world spam email dataset

Text preprocessing: cleaning, stopword removal, lemmatization

One-hot encoding and sequence padding

Pre-trained model loading using Pickle

Class-based, modular Python design

Logging and exception handling

🛠 **Technologies & Libraries Used**

Python 3

Pandas & NumPy – Data processing

NLTK – Text preprocessing and lemmatization

TensorFlow / Keras – Model inference

Scikit-learn – Supporting utilities

Pickle – Model serialization

🧠 **How It Works**

Loads the spam email dataset

Cleans the email text (lowercase, punctuation removal)

Removes stopwords and applies lemmatization

Converts text into numerical form using one-hot encoding

Pads sequences to a fixed length

Loads the trained spam classification model

Predicts whether the email is Spam or Ham

📂 **Project Structure**

spam-mail-classifier/

├── main.py                 # Main application file

├── spam_detection.pkl      # Trained spam classifier model

├── spam.csv                # Dataset file

├── log.py                  # Logging configuration

├── README.md               # Project documentation

▶️ How to Run the Project

1️⃣ Install Required Libraries

pip install numpy pandas nltk tensorflow scikit-learn

2️⃣ Download NLTK Resources

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords

3️⃣ Run the Application

python main.py

📌 **Sample Input**

email = "Congratulations! You have won a free lottery prize. Click here now"

📄 **Sample Output**

Detection of the mail: Spam

🎯 **Use Cases**

Email spam filtering systems

Message moderation tools

Cybersecurity and fraud detection

NLP learning and experimentation

📈 **Learning Outcomes**

Building NLP preprocessing pipelines

Understanding text classification

Working with real-world email data

Model inference using TensorFlow

Applying lemmatization and stopword removal

🔮 **Future Enhancements**

Train the model within the project

Add multi-class email categorization

Build a web interface using Flask or Streamlit

Improve accuracy using LSTM / Bi-LSTM models

🤝 **Contributing**

Contributions, suggestions, and improvements are welcome!

📬 **Contact**

Name: P.Naveen Kumar

🔗 LinkedIn: www.linkedin.com/in/naveenkumar-puppala-b87737332

🐙 Gmail: puppalanaveenkumar11@gmail.com
