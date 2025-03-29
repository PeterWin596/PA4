#!/usr/bin/env python3

"""
scorer.py

Compares system-generated sense predictions with gold standard answers,
and prints overall accuracy and a confusion matrix.

Usage:
    python3 scorer.py my-line-answers.txt line-key.txt
"""

import sys
import re
from collections import defaultdict

def load_answers(filename):
    answers = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Match lines like: <answer instance="line-n.w8_059:8174:" senseid="phone"/>
            match = re.search(r'instance="(.*?)"\s+senseid="(.*?)"', line)
            if match:
                instance_id = match.group(1)
                sense = match.group(2)
                answers[instance_id] = sense
    return answers

def load_predictions(filename):
    predictions = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                predictions[parts[0]] = parts[1]
    return predictions

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scorer.py my-line-answers.txt line-key.txt", file=sys.stderr)
        sys.exit(1)

    predictions = load_predictions(sys.argv[1])
    gold = load_answers(sys.argv[2])

    print(f"Loaded {len(gold)} gold answers.", file=sys.stderr)

    total = 0
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))

    for instance_id, gold_label in gold.items():
        predicted_label = predictions.get(instance_id, "NONE")
        if predicted_label == gold_label:
            correct += 1
        confusion[gold_label][predicted_label] += 1
        total += 1

    if total == 0:
        print("Error: No valid gold answers loaded. Cannot compute accuracy.", file=sys.stderr)
        sys.exit(1)

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%\n")

    print("Confusion Matrix:")
    labels = sorted(set(gold.values()) | set(predictions.values()))
    print("Gold\\Pred", "\t".join(labels))
    for g in labels:
        row = [str(confusion[g][p]) for p in labels]
        print(g, "\t".join(row))

if __name__ == "__main__":
    main()
