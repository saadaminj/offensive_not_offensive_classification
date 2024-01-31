from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from text_classification_model import TextClassificationModel, ClassifierAdapter

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('labeled_data.csv')
X = df['tweet'].apply(TextClassificationModel().preprocess_tweet)
X_vectorized, vectorizer = TextClassificationModel().vectorize_text(X)
df['class'] = df['class'].replace({0: 1})
df['class'] = df['class'].replace({2: 0})
y = df['class']

# Train models
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

lr_model = ClassifierAdapter(LogisticRegression)
svc_model = ClassifierAdapter(SVC, kernel='linear', C=1.0)

# lr_model.train_and_eval_model(X_train, X_test, y_train, y_test, save_model=True)
# svc_model.train_and_eval_model(X_train, X_test, y_train, y_test, save_model=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_wordcloud/<int:target>')
def generate_wordcloud(target):
    offensive_tweets = X[y == target]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(offensive_tweets))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Non-Offensive Tweets') if target == 0 else plt.title('Word Cloud for Offensive Tweets')
    plt.axis('off')
    plt.savefig('static/wordcloud.png')
    return render_template('wordcloud.html')

@app.route('/train_model')
def train_model():
    lr_model = ClassifierAdapter(LogisticRegression)
    svc_model = ClassifierAdapter(SVC, kernel='linear', C=1.0)

    lr_model.train_and_eval_model(X_train, X_test, y_train, y_test, save_model=True, force_train=True)
    svc_model.train_and_eval_model(X_train, X_test, y_train, y_test, save_model=True, force_train=True)

    return "Models trained and saved successfully!"

@app.route('/predict_tweet', methods=['POST'])
def predict_tweet():
    tweet = request.form['tweet']
    new_tweet_preprocessed = TextClassificationModel().preprocess_tweet(tweet)
    new_tweet_vectorized = vectorizer.transform([new_tweet_preprocessed])

    lr_prediction = lr_model.predict_new_tweet(new_tweet_vectorized)
    svc_prediction = svc_model.predict_new_tweet(new_tweet_vectorized)

    return render_template('index.html', tweet= tweet, lr_prediction=lr_prediction, svc_prediction=svc_prediction)

if __name__ == '__main__':
    app.run(debug=True)
