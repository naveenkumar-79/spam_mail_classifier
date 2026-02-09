import numpy as np
import pandas as pd
import sys
import sklearn
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
nltk.download('punkt',quiet=True)
nltk.download('wordnet')
nltk.download('stopwords')
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from log import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings('ignore')
class spam_analysis:
    def __init__(self):
        self.df=pd.read_csv(r'D:\NLP_projects\spam_classifier\spam.csv',encoding='latin_1')
        logger.info(self.df.head(5))
    def model_loading(self):
        try:
            with open('spam_detection.pkl', 'rb') as f:
                self.m = pickle.load(f)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Performance Error : {er_lin.tb_lineno} : due to {er_msg}")
    def model_testing(self):
        labels = ['positive', 'negative']
        dic_size = 5500
        review='Even my brother is not like to speak with me. They treat me like aids patent.'
        text = review[0].lower()
        text = ''.join([i for i in text if i not in string.punctuation])
        text = ' '.join([lemma.lemmatize(i) for i in text.split() if i not in stopwords.words('english')])
        v = [one_hot(i, dic_size) for i in [text]]
        p = pad_sequences(v, maxlen=80, padding='post')
        logger.info(f'Detection of the mail: {(labels[np.argmax(self.m.predict(p))])}')

if __name__ == "__main__":
    obj=spam_analysis()
    obj.model_loading()
    obj.model_testing()