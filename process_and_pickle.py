import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer

## Load and parse Q&A
def load_qa(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    questions, answers = [], []
    current_questions = []
    current_answer = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q;"):
            current_questions.append(line[2:].strip())
        elif line.startswith("A;"):
            current_answer = line[2:].strip()
            for q in current_questions:
                questions.append(q)
                answers.append(current_answer)
            current_questions = []
            current_answer = ""
    return questions, answers

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

## Load Q&A and preprocess
questions, answers = load_qa("data.txt")
questions = [preprocess(q) for q in questions]

## Fit TF-IDF
vectorizer = TfidfVectorizer()
q_vectors = vectorizer.fit_transform(questions)

## Save for serverless use
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("q_vectors.pkl", "wb") as f:
    pickle.dump(q_vectors, f)

with open("answers.pkl", "wb") as f:
    pickle.dump(answers, f)

print("Preprocessing done. Pickle files created.")
