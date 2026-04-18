# Detecting Fake News Using Logistic Regression and NLP

This project is a simple machine learning application that classifies news as **Fake or Real** using basic NLP techniques.

## About the Project

The main idea is to take a news headline or article and predict whether it is fake or real. The model is trained on a dataset of labeled news articles and uses text processing techniques to make predictions.

## Features

* Uses Logistic Regression and Naive Bayes models
* TF-IDF is used for text vectorization
* Built using Streamlit for a simple user interface
* Shows prediction along with confidence score
* Works with both short and long text input

## Models Used

* Logistic Regression
* Multinomial Naive Bayes

## How it Works

1. User enters news text
2. Text is cleaned and processed
3. Converted into numerical form using TF-IDF
4. Both models make predictions
5. Final result is displayed

## Important Note

The model is trained on a specific dataset of fake and real news.

* Results may not always be accurate for real-world news
* Short or unusual text may give inconsistent results
* This project is mainly for learning and demonstration

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

## Requirements

* streamlit
* pandas
* scikit-learn

## Dataset

The dataset is not included in this repository because of its size.

You can download it from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, place these files in the project folder:

* Fake.csv
* True.csv

## Model Files

The trained model files are also not included.

To generate them, run:

python train.py

This will create:

* model_logistic.pkl
* model_nb.pkl
* vectorizer.pkl

After that, you can run the app normally.



