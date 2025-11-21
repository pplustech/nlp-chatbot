from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import string

## -----------------------------
## Preprocess text
## -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

## -----------------------------
## Load precomputed data
## -----------------------------
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("q_vectors.pkl", "rb") as f:
    q_vectors = pickle.load(f)

with open("answers.pkl", "rb") as f:
    answers = pickle.load(f)

## -----------------------------
## Chatbot class
## -----------------------------
class BATELECChatbot:
    def __init__(self, vectorizer, q_vectors, answers):
        self.vectorizer = vectorizer
        self.q_vectors = q_vectors
        self.answers = answers

    def get_response(self, user_input):
        user_input_processed = preprocess(user_input)
        user_vec = self.vectorizer.transform([user_input_processed])
        similarity_scores = cosine_similarity(user_vec, self.q_vectors)
        max_index = similarity_scores.argmax()
        if similarity_scores[0, max_index] < 0.2:
            return "Sorry, I don't have an answer for that. Can you rephrase?"
        return self.answers[max_index]

## -----------------------------
## Initialize FastAPI
## -----------------------------
app = FastAPI(title="BATELEC Chatbot API")
chatbot = BATELECChatbot(vectorizer, q_vectors, answers)
handler = Mangum(app)

class Query(BaseModel):
    message: str

@app.post("/chat")
def chat(query: Query):
    response = chatbot.get_response(query.message)
    return {"response": response}
