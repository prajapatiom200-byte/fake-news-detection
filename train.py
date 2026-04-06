import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import pickle

from utils import preprocess_text

# ----------------------------
# 1. LOAD DATA
# ----------------------------
data = pd.read_csv("data.csv")

# ----------------------------
# 2. FIX: USE TITLE + TEXT
# ----------------------------
if "title" in data.columns:
    data["content"] = data["title"] + " " + data["text"]
else:
    data["content"] = data["text"]

# ----------------------------
# 3. CLEAN TEXT
# ----------------------------
data["content"] = data["content"].astype(str).apply(preprocess_text)

# ----------------------------
# 4. HANDLE CLASS IMBALANCE
# ----------------------------
real = data[data.label == 1]
fake = data[data.label == 0]

if len(real) > len(fake):
    fake = resample(fake, replace=True, n_samples=len(real), random_state=42)
else:
    real = resample(real, replace=True, n_samples=len(fake), random_state=42)

data = pd.concat([real, fake])

# ----------------------------
# 5. FEATURES & LABELS
# ----------------------------
X = data["content"]
y = data["label"]

# ----------------------------
# 6. TF-IDF (IMPROVED)
# ----------------------------
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),   # VERY IMPORTANT
    min_df=2
)

X = vectorizer.fit_transform(X)

# ----------------------------
# 7. TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 8. MODEL 1: LOGISTIC REGRESSION
# ----------------------------
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# ----------------------------
# 9. MODEL 2: NAIVE BAYES
# ----------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# ----------------------------
# 10. EVALUATION
# ----------------------------
def evaluate(y_test, y_pred, name):
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

evaluate(y_test, y_pred_log, "Logistic Regression")
evaluate(y_test, y_pred_nb, "Naive Bayes")

# ----------------------------
# 11. SAVE MODELS
# ----------------------------
pickle.dump(log_model, open("model_logistic.pkl", "wb"))
pickle.dump(nb_model, open("model_nb.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Models trained and saved successfully")