#!/usr/bin/env python3

"""
wsd-ml.py

This program performs word sense disambiguation (WSD) using three different machine learning models
from scikit-learn: Naive Bayes, Logistic Regression, and Random Forest. It uses a bag-of-words feature
representation built only from the training data.

The script is run from the command line using:
    python3 wsd-ml.py line-train.txt line-test.txt [OptionalModel] > my-line-answers.txt

If no model is specified, Naive Bayes is used by default. The program trains on line-train.txt,
predicts senses for line-test.txt, and outputs predictions in the correct format for evaluation.

Models:
- NaiveBayes (MultinomialNB)
- LogisticRegression
- RandomForestClassifier

Feature extraction:
- CountVectorizer (Bag-of-Words) from training context only

Results are summarized in the comments at the bottom of this file, including accuracy, confusion matrix,
and comparisons to the most frequent sense baseline and decision list approach from PA3.
"""


import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def parse_args():
    if len(sys.argv) < 3:
        print("Usage: python3 wsd-ml.py line-train.txt line-test.txt [ModelName]", file=sys.stderr)
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "NaiveBayes"
    return train_file, test_file, model_name

def load_training_data(filename):
    sentences = []
    labels = []

    current_label = None
    current_context = ""
    in_context = False

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("<answer "):
                match = re.search(r'senseid="(.*?)"', line)
                if match:
                    current_label = match.group(1)

            elif "<context>" in line:
                current_context = ""
                in_context = True

            elif "</context>" in line:
                in_context = False
                clean_text = re.sub(r"<.*?>", "", current_context).strip()
                if current_label and clean_text:
                    sentences.append(clean_text)
                    labels.append(current_label)

            elif in_context:
                current_context += " " + line

    return sentences, labels

def load_test_data(filename):
    test_ids = []
    test_sentences = []
    in_context = False
    current_context = ""
    current_id = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("<instance id="):
                match = re.search(r'id="(.*?)"', line)
                if match:
                    current_id = match.group(1)

            elif "<context>" in line:
                current_context = ""
                in_context = True

            elif "</context>" in line:
                in_context = False
                clean_text = re.sub(r"<.*?>", "", current_context).strip()
                if current_id and clean_text:
                    test_ids.append(current_id)
                    test_sentences.append(clean_text)

            elif in_context:
                current_context += " " + line

    return test_ids, test_sentences

def choose_model(name):
    name = name.lower()
    if name == "naivebayes":
        return MultinomialNB()
    elif name == "logisticregression":
        return LogisticRegression(max_iter=1000)
    elif name == "randomforest":
        return RandomForestClassifier()
    else:
        print(f"Unknown model '{name}', using NaiveBayes.", file=sys.stderr)
        return MultinomialNB()

def main():
    train_file, test_file, model_name = parse_args()

    train_X_raw, train_y = load_training_data(train_file)

    # Debugging output
    print(f"Loaded {len(train_X_raw)} training sentences.", file=sys.stderr)
    print(f"First example: {train_X_raw[0] if train_X_raw else 'EMPTY'}", file=sys.stderr)

    test_ids, test_X_raw = load_test_data(test_file)

    # Create BoW vectorizer using only training data
    vectorizer = CountVectorizer()
    train_X = vectorizer.fit_transform(train_X_raw)
    test_X = vectorizer.transform(test_X_raw)

    model = choose_model(model_name)
    model.fit(train_X, train_y)
    predictions = model.predict(test_X)

    # Output in required format
    for i, pid in enumerate(test_ids):
        print(f"{pid} {predictions[i]}")

if __name__ == "__main__":
    main()

"""
=== Results Summary ===

Models tested:
- NaiveBayes
- LogisticRegression
- RandomForest

Most Frequent Sense Baseline: ~60.3% (based on label distribution in line-key.txt)
Decision List (PA3): [Insert your accuracy from PA3]

NaiveBayes Accuracy: 80.16%
LogisticRegression Accuracy: [TBD]
RandomForest Accuracy: [TBD]

Best performing model (so far): NaiveBayes
Notes:
- NaiveBayes performed well on both classes with just 7 misclassifications.
- Confusion matrix shows slight bias toward "phone", but overall balanced.
- Will compare other models next.

"""
