"""
Bangla Health Chatbot - IEEE BECITHCON-2026
Backend: Flask + Groq (LLaMA 3.3 70B) + trained TF-IDF category classifier
Author: Shibli Sanjid Faheem & Team
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os, json, datetime, uuid
from dotenv import load_dotenv

from nlp_engine import predict_category, retrieve_similar, CATEGORY_LABELS_BN

load_dotenv()
import sys
print("Starting...", flush=True)
print(f"KEY: {bool(os.getenv('GROQ_API_KEY'))}", flush=True)

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file!")

client = Groq(api_key=API_KEY)

# llama-3.1-8b-instant garbles Bangla script ("????") on long prompts;
# the 70B model produces clean Bangla.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """tumi ekjon bangla bhasay kotha bola swasthoseba sahakari.
Tomaar kaj holo gramer sadharon manusder tader saririk lokkhon shune sambhab karon ebong prathamik poramorso deowa.

Kothor niyomsomuh:
1. Sobsomoy shudhumatro Banglay uttor dao
2. Sohoj, sadharon Bangla byabohar koro
3. Kokhono nischitbhabe rog nirdharon korbe na
4. Protiti uttor oboshshoi nicer tinti bhage dao:

Sambhab Karon:
(ekhane 1-3 ta sambhab karon likhao)

Ghore ja korben:
(ekhane 2-3 ta sohoj ghoroa poramorso dao)

Kokhon Doctor Dekhaben:
(ekhane sposhtobbhabe bolo kon lokkhone obosshoi doctor er kache jete hobe)

5. Uttor sonkhipto o porishkar rakho
6. Rogir proti sohanubhutishil o sommanjonok bhasay kotha bolo"""

def build_dataset_context(category, similar_examples):
    """Extra system context from the trained classifier + dataset retrieval."""
    lines = []
    if category:
        lines.append(
            f"Amader trained classifier onujayi byabaharkarir ei bartar bibhag: '{category}'."
        )
        if category == "emergency":
            lines.append(
                "Eta EMERGENCY hote pare! Uttorer ekdom shurute sposhto kore bolo: "
                "EKHONI nikotostho hospital er jorurii bibhag ba doctor er kache jete hobe."
            )
    if similar_examples:
        lines.append(
            "Nicher udahoron gulo amader Bangla health dataset theke newa, "
            "ei prosner sathe egulor sob cheye beshi mil ache. Tomar poramorsho "
            "jotota sombhob ei udahoron gulor dhoron o poramorsher sathe mil rekhe dao, "
            "kintu upore dewa format ar niyom (kokhono nischit rog nirdharon na kora) "
            "obosshoi mene cholo:"
        )
        for i, ex in enumerate(similar_examples, 1):
            lines.append(f"{i}. [{ex['category']}] {ex['text'][:400]}")
    return "\n".join(lines)


sessions = {}
LOG_FILE = "session_logs.json"

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_log(entry):
    logs = load_logs()
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/new-session", methods=["POST"])
def new_session():
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = []
    return jsonify({"session_id": session_id})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    history = data.get("history", [])

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        category, confidence = predict_category(user_msg)
        similar = retrieve_similar(user_msg, k=3)

        system_content = SYSTEM_PROMPT
        dataset_context = build_dataset_context(category, similar)
        if dataset_context:
            system_content += "\n\n" + dataset_context

        messages = [{"role": "system", "content": system_content}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": user_msg})

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        bot_reply = completion.choices[0].message.content

        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": user_msg,
            "bot_response": bot_reply,
            "input_length": len(user_msg),
            "response_length": len(bot_reply),
            "predicted_category": category,
            "confidence": round(confidence, 3)
        }
        save_log(log_entry)

        return jsonify({
            "reply": bot_reply,
            "session_id": session_id,
            "category": category,
            "category_bn": CATEGORY_LABELS_BN.get(category),
            "confidence": round(confidence, 3),
            "emergency": category == "emergency"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export-log")
def export_log():
    logs = load_logs()
    return jsonify({"total_sessions": len(logs), "logs": logs})

@app.route("/stats")
def stats():
    logs = load_logs()
    return jsonify({"total_conversations": len(logs), "last_updated": datetime.datetime.now().isoformat()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
