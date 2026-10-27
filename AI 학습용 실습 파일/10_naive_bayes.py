"""10. 나이브 베이즈 (Naive Bayes) - 텍스트 분류 (뉴스 그룹 대신 간단한 한글/영문 문장)"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "machine learning improves factory quality", "deep learning predicts equipment failure",
    "neural network detects defects", "sensor data analysis with AI",
    "the football team won the match", "the player scored a goal",
    "the coach announced the season schedule", "fans cheered in the stadium",
]
labels = ["tech"] * 4 + ["sports"] * 4

model = make_pipeline(CountVectorizer(), MultinomialNB()).fit(texts, labels)

tests = ["AI model for predictive maintenance", "the team scored in the final match"]
for t, p in zip(tests, model.predict(tests)):
    print(f"'{t}' -> {p}")
