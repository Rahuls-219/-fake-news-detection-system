# 🔍 Fake News Detection System
> Real-Time · Explainable · Offline-First News Verification  
> Built for **FusionX Hackathon 2026** by **CodeTrio** — Presidency University, Bengaluru

---

## 🎯 What It Does
- Accepts any news headline or article text
- Classifies it as **FAKE** or **REAL** in under 1 second
- Shows **confidence score** with visual progress bar
- Highlights **top signal keywords** driving the prediction
- Generates a **human-readable insight** explaining the result
- Supports **Light & Dark mode** toggle

---

## 🧠 Tech Stack
| Component | Technology |
|-----------|-----------|
| Model | Logistic Regression |
| Features | TF-IDF (unigrams + bigrams) |
| UI | Streamlit |
| Runtime | Python 3.9+ |
| Deployment | Streamlit Cloud |

**No deep learning. No external APIs. Fully offline.**

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_dataset.py
```

### 3. Train Model
```bash
python train_model.py
```

### 4. Launch App
```bash
streamlit run app.py
```

Open → http://localhost:8501

---

## 📁 Project Structure
```
fake_news_detection/
├── app.py                  # Streamlit UI (main entry point)
├── train_model.py          # Model training script
├── utils.py                # Prediction, cleaning, explanation logic
├── generate_dataset.py     # Synthetic dataset generator
├── requirements.txt        # Python dependencies
├── data/
│   └── news_dataset.csv    # Training data (generated)
└── model/                  # Auto-created after training
    ├── logistic_model.pkl
    └── tfidf_vectorizer.pkl
```

---

## 🧪 Test Inputs

### 🚨 Fake News Examples
```
SHOCKING: Scientists discover miracle cure for all diseases hidden by Big Pharma for decades. Share before deleted!

BREAKING: Government secretly putting mind control chemicals in drinking water, whistleblower reveals truth!
```

### ✅ Real News Examples
```
Federal Reserve raises interest rates by 25 basis points amid ongoing inflation concerns, officials confirmed.

University researchers develop new battery technology that charges 40 percent faster, study published in Nature.
```

---

## 🌐 Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo
4. Set main file: `app.py`
5. Click **Deploy** → Get your live link!

> **Note:** Add a `setup.sh` if you need to pre-run training on the cloud.

---

## 👥 Team
**CodeTrio** — FusionX Hackathon 2026 · ID: FXH26-IP-BDDS-006  
- Sujay Babli (Captain)
- Vishalkumar
- Vikas Gouda BH
- Rahul

Dept. of Computer Science and Engineering  
Presidency University, Bengaluru
