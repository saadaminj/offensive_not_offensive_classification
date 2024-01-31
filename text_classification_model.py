from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class TextClassificationModel:
    def __init__(self, classifier=None):
        self.classifier = classifier

    def preprocess_tweet(self, tweet):
        tweet = re.sub(r"@[A-Za-z0-9]+", "", tweet)  # Remove mentions
        tweet = re.sub(r"#", "", tweet)  # Remove hashtags
        tweet = re.sub(r"RT[\s]+", "", tweet)  # Remove retweets
        tweet = re.sub(r"https?:\/\/\S+", "", tweet)  # Remove URLs
        tweet = tweet.lower()  # Convert to lowercase
        stop_words = set(stopwords.words('english'))
        tweet = ' '.join(word for word in word_tokenize(tweet) if word.isalnum() and word not in stop_words)  # Tokenization and removal of stopwords
        tweet = re.sub(r'[^\w\s]', '', tweet)  # Removing punctuation
        tweet = tweet.strip()  # Remove extra whitespaces
        return tweet

    def vectorize_text(self, X):
        # Text Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        X_vectorized = vectorizer.fit_transform(X)
        return X_vectorized, vectorizer

    def train_and_eval_model(self, X_train, X_test, y_train, y_test, save_model=False, force_train = False):
        if force_train:        
            self.classifier.fit(X_train, y_train)
            # Model Evaluation
            y_pred = self.classifier.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))

            if save_model:
                self.save_model()
        else:
            try:
                self.load_model()
            except:
                self.train_and_eval_model(X_train, X_test, y_train, y_test, save_model=True)

    def save_model(self):
        with open(f'models/model_{type(self.classifier).__name__}.pkl', 'wb') as model_file:
            pickle.dump(self.classifier, model_file)
            
    def load_model(self):
        with open(f'models/model_{type(self.classifier).__name__}.pkl', 'rb') as model_file:
            self.classifier = pickle.load(model_file)
        
    def predict_new_tweet(self, new_tweet):
        self.load_model()
        prediction = self.classifier.predict(new_tweet)
        print(f'model : {type(self.classifier).__name__} , prediction : {prediction[0]}')
        return prediction[0]


class ClassifierAdapter(TextClassificationModel):
    def __init__(self, classifier_class, **kwargs):
        classifier_instance = classifier_class(**kwargs)
        super().__init__(classifier_instance)