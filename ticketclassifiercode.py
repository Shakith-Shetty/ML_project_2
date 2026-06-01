# ==========================
# IMPORT LIBRARIES
# ==========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# UPLOAD DATASET
# ==========================
from google.colab import files
uploaded = files.upload()

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("all_tickets.csv")

# ==========================
# CHECK DATASET
# ==========================

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTopic Groups:")
print(df['Topic_group'].value_counts())

# ==========================
# DEFINE FEATURES & TARGET
# ==========================

X = df['Document']

y = df['Topic_group']

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# BUILD MODEL
# ==========================

model = Pipeline([

    ('tfidf', TfidfVectorizer(
        stop_words='english'
    )),

    ('classifier', LinearSVC())

])

# ==========================
# TRAIN MODEL
# ==========================

model.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# TEST CUSTOM TICKET
# ==========================

sample_ticket = [
    "Laptop is not connecting to office VPN"
]

prediction = model.predict(sample_ticket)

print("\nPredicted Category:")
print(prediction[0])


