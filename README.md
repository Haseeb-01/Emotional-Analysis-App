
# 🎭 Emotion Analysis System with DistilBERT & Streamlit

Welcome to the **Emotion Analysis System**!  
This project leverages a fine-tuned `DistilBERT` model to analyze emotional tones in text, such as customer feedback or social media posts. It supports **multi-label emotion classification** for 28 emotions like `joy`, `anger`, `gratitude`, etc.

> 🧠 Powered by **Transformers**, deployed with **Streamlit**, and tunneled through **ngrok** for instant web access.  

---

## 🌟 Project Highlights

✅ **Multi-Label Emotion Detection**  
🔍 Predicts **28 emotions** using `DistilBERT` on the **GoEmotions** dataset.

⚡ **Real-Time Analysis**  
🧾 Input text → Get **emotion predictions + confidence scores** instantly.

🖥️ **Interactive Web Interface**  
Built with **Streamlit**, hosted via **ngrok** for seamless accessibility.

📦 **Easy Cloud Deployment**  
Runs smoothly in **Google Colab** with just a few commands.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit app for real-time emotion analysis |
| `requirements.txt` | Required Python libraries |
| `README.md` | You're reading it! 📖 |

> ⚠️ **Note**: The model file `distilbert_emotion_model.pt` is too large for GitHub. It’s **hosted on Google Drive** and downloaded by the app during runtime.

---

## 🚀 How to Run (Google Colab + Streamlit + ngrok)

### 1. Clone the Repo
```bash
!git clone https://github.com/Haseeb-01/emotion-analysis-app.git
%cd emotion-analysis-app
```

### 2. Install Dependencies
```bash
!pip install -r requirements.txt
!pip install pyngrok
```

### 3. Run the Streamlit App via ngrok
```python
from pyngrok import ngrok
!ngrok authtoken YOUR_NGROK_AUTHTOKEN
get_ipython().system_raw('streamlit run app.py --server.port 8501 &')
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url}")
```

🔗 **Screenshots live testing the app through the ngrok**Soon you can access the app.
<img width="344" alt="screenshot2" src="https://github.com/user-attachments/assets/019ef3b1-3685-475f-8c7f-5bf2751e2477" />
<img width="348" alt="screenshot3" src="https://github.com/user-attachments/assets/718ef6ac-7850-4724-bbc9-9fbdf6fc6704" />
<img width="336" alt="screenshot4" src="https://github.com/user-attachments/assets/38f3aef3-771b-4927-9a6f-cc68ba2004d1" />

---

## 🧪 Example Inputs

```text
"I’m so happy with this product, it’s amazing!"
→ joy: 0.852, admiration: 0.673, excitement: 0.512

"This service is terrible, I’m so disappointed."
→ disappointment: 0.789, anger: 0.645
```

---

## 🔧 Prerequisites

- **Dataset**: [GoEmotions](https://github.com/google-research/googleresearch/tree/master/goemotions) (via Kaggle)
- **Model**: Fine-tuned `DistilBERT` model (auto-download from Google Drive)
- **Dependencies**: Listed in `requirements.txt`
  - `streamlit`, `transformers`, `torch`, `numpy`, `gdown`
- **ngrok**: Sign up at [ngrok.com](https://ngrok.com) and get your auth token

---

## 💡 Future Enhancements

- ✅ Batch processing of multiple texts
- 📊 Emotion distribution charts and visualizations
- 🔗 Integration with social media APIs for real-time feedback monitoring

---

## 📓 Open in Google Colab
  🚀 [Click here to run this project in Google Colab]
  (https://colab.research.google.com/drive/122rvJfgUwNPEuSDQUGMdc0KdV8varX6E?usp=sharing)

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Credits

Developed with ❤️ by **Haseeb Cheema**  
Special thanks for all the guidance and support!

---

## 🌐 Access the App

🔗 **Streamlit App URL**:  
Soon you can access the app.
```
