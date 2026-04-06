# 📰 ML-Based Fake News Detection using NLP

This project detects whether a news article is **Fake or Real** using Natural Language Processing (NLP) and Machine Learning models.

## 🚀 Features

* Logistic Regression & Naive Bayes models
* TF-IDF vectorization
* Streamlit interactive UI
* Confidence scores for predictions
* Works on short and long news text

## 🧠 Models Used

* Logistic Regression
* Multinomial Naive Bayes

## 📊 How it Works

1. Input news text
2. Preprocess (clean + normalize)
3. Convert using TF-IDF
4. Predict using ML models
5. Compare model outputs
6. Show final verdict

## ⚠️ Important Note

This model is trained on a specific dataset of fake and real news.

* Results may vary for real-world or unseen news
* Performance may differ for short, informal, or trending content
* This project is intended for **educational and demonstration purposes only**

## 🖥️ Run Locally

pip install -r requirements.txt
streamlit run app.py

## 📦 Requirements

* streamlit
* pandas
* scikit-learn
## 📂 Dataset

The dataset used for training is not included in this repository due to size limitations.

You can download it from the following sources:

* Fake News Dataset (Kaggle): https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading:

1. Extract the files
2. Place them in the project folder as:

   * Fake.csv
   * True.csv

## ⚙️ Model Files

Pre-trained model files (`.pkl`) are not included in this repository.

To generate them:

```bash
python train.py
```

This will create:

* model_logistic.pkl
* model_nb.pkl
* vectorizer.pkl

After that, run:

```bash
streamlit run app.py
```


