from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI(title="Sentiment Analysis API")

print("Loading model... this happens once at startup")
analyzer = SentimentIntensityAnalyzer()
print("Model loaded successfully!")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: TextInput):
    scores = analyzer.polarity_scores(input.text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "text": input.text,
        "label": label,
        "confidence": round(abs(compound), 4)
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
