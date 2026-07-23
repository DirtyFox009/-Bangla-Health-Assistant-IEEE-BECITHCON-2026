# বাংলা স্বাস্থ্য সহকারী 🏥
A Bangla-language AI health chatbot for rural communities.
Built for IEEE BECITHCON-2026.

## How to Run
1. Clone this repo
2. Get a free API key from https://console.groq.com
3. Create a .env file: GROQ_API_KEY=your_key_here
4. pip install -r requirements.txt
5. python train_model.py  (trains the health-category classifier on data/bangla_health_dataset.csv — only needed once; pre-trained artifacts in model/ are committed so deployment can skip this)
6. python app.py
7. Open http://localhost:5000

## Trained Model
- Dataset: `data/bangla_health_dataset.csv` — 1,800 Bangla/Banglish health texts, 6 categories (symptoms, doctor, medicine, nutrition, fever, emergency)
- Model: TF-IDF (word + char n-grams) + Logistic Regression (scikit-learn)
- Every chat message is classified; the category is shown in the UI, emergency messages trigger an urgent alert, and the most similar dataset entries are given to the LLM as reference answers
- Evaluation metrics (accuracy, per-class F1, confusion matrix) are written to `model/metrics.json` by `train_model.py`

## To access the Live link
-LInk:https://bangla-health-assistant-ieee-becithcon.onrender.com/
