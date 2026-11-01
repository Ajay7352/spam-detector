from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="SMS Spam Checker", page_icon="🛡️")
st.title("🛡️ SMS Spam Checker")
st.write("Paste an SMS message to estimate whether it may be spam.")

data = pd.read_csv(Path(__file__).parent / "data" / "spam.csv")
data = data[["label", "message"]].dropna().drop_duplicates()
data["label"] = data["label"].str.lower().str.strip()

x_train, x_test, y_train, y_test = train_test_split(
    data["message"], data["label"],
    test_size=0.2, random_state=42, stratify=data["label"]
)

model = Pipeline([
    ("words", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])
model.fit(x_train, y_train)

accuracy = accuracy_score(y_test, model.predict(x_test))
st.caption(f"Test-set accuracy: {accuracy:.2%}")

message = st.text_area("SMS message", placeholder="Type or paste a message...")
if st.button("Check message"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        result = model.predict([message])[0]
        spam_index = list(model.classes_).index("spam")
        spam_probability = model.predict_proba([message])[0][spam_index]

        if result == "spam":
            st.error(f"🚨 Likely SPAM — estimated spam probability: {spam_probability:.1%}")
        else:
            st.success(f"✅ Likely HAM — estimated spam probability: {spam_probability:.1%}")

st.info("This is a model estimate, not a guarantee. Be careful with suspicious links.")
