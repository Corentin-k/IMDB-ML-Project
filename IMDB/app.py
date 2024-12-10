from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import joblib

# Load the trained model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize FastAPI
app = FastAPI()

origins = [
    "http://localhost:8080",
"http://localhost:8081",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)



# Request body for prediction
class Review(BaseModel):
    review: str


@app.get("/")
def read():
    return {"message": "Welcome to the Sentiment Analysis API!"}


@app.post("/predict/")
def predict_sentiment(review: Review):
    # Transform the review using the vectorizer
    transformed_review = vectorizer.transform([review.review])

    # Make prediction
    sentiment = model.predict(transformed_review)[0]
    probabilities = model.predict_proba(transformed_review)[0]

    # Return the sentiment and probabilities
    return {
        "sentiment": sentiment,
        "probability": {
            "negative": round(probabilities[0], 4),
            "positive": round(probabilities[1], 4)
        }
    }
