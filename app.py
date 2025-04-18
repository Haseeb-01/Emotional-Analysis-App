import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

# --------------------- UI Setup ---------------------
st.set_page_config(page_title="🎭 Emotion Analyzer", layout="centered")
st.title("🎭 Emotion Analyzer")
st.markdown("**Understand the emotional undertone behind text instantly.**")
st.markdown("Powered by **DistilBERT** and fine-tuned on the GoEmotions dataset.")

st.markdown("---")

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=28, problem_type="multi_label_classification"
        )
        model.load_state_dict(torch.load("/content/distilbert_emotion_model.pt", map_location=torch.device("cpu")))
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_model()
if tokenizer is None or model is None:
    st.stop()

# --------------------- Emotion Labels ---------------------
label_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# --------------------- Prediction Function ---------------------
def predict_emotions(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    threshold = 0.3
    predicted_labels = (probs > threshold).astype(int)
    results = {label: prob for label, prob, pred in zip(label_columns, probs, predicted_labels) if pred == 1}
    return results

# --------------------- Input Section ---------------------
st.subheader("📝 Enter Text")
user_input = st.text_area("Type something emotional...", placeholder="e.g., I can't believe how amazing this is!", height=100)

# --------------------- Emotion Analysis ---------------------
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Detecting emotions..."):
            try:
                emotion_results = predict_emotions(user_input)
                if emotion_results:
                    st.success("✅ Emotions Detected:")

                    sorted_emotions = sorted(emotion_results.items(), key=lambda x: x[1], reverse=True)

                    for label, score in sorted_emotions:
                        st.markdown(f"**{label.capitalize()}**: {score:.3f}")
                        st.progress(float(score))

                else:
                    st.info("🤔 No strong emotions detected above the threshold.")
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

# --------------------- Examples Section ---------------------
with st.expander("📘 Example Inputs"):
    st.markdown("- _I love this product, it's amazing!_")
    st.markdown("- _This service is terrible, I'm so disappointed._")
    st.markdown("- _Thank you so much, I really appreciate the help._")

# --------------------- Footer ---------------------
st.markdown("---")
st.markdown("🔧 **Model**: DistilBERT fine-tuned for multi-label emotion classification.  \n💡 **Dataset**: [GoEmotions (by Google)](https://github.com/google-research/google-research/tree/master/goemotions)  \n🚀 **Built with**: Streamlit + Transformers")

st.markdown("💡 **Credit goes to Muhammad Haseeb for making this app.**")
