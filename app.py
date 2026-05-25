"""
Bangla Health Chatbot - IEEE BECITHCON-2026
Backend: Flask + Groq (LLaMA 3.1)
Author: Shibli Sanjid Faheem & Team
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os, json, datetime, uuid
from dotenv import load_dotenv

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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": user_msg})

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
            "response_length": len(bot_reply)
        }
        save_log(log_entry)

        return jsonify({"reply": bot_reply, "session_id": session_id})

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
