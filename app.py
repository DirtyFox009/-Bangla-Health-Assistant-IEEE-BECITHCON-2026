"""
Bangla Health Chatbot — IEEE BECITHCON-2026
Backend: Flask + Groq (LLaMA 3.1)
Author: Shibli Sanjid Faheem & Team
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os, json, datetime, uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ─── Configure Groq ─────────────────────────────────────────────
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file!")

client = Groq(api_key=API_KEY)

# ─── System Prompt (Bangla) ─────────────────────────────────────
SYSTEM_PROMPT = """তুমি একজন বাংলা ভাষায় কথা বলা প্রাথমিক স্বাস্থ্যসেবা সহকারী।
তোমার কাজ হলো গ্রামের সাধারণ মানুষদের তাদের শারীরিক লক্ষণ শুনে সম্ভাব্য কারণ এবং প্রাথমিক পরামর্শ দেওয়া।

কঠোর নিয়মসমূহ:
১. সবসময় শুধুমাত্র বাংলায় উত্তর দাও — কোনো ইংরেজি শব্দ ব্যবহার করবে না
২. সহজ, সাধারণ বাংলা ব্যবহার করো যা গ্রামের মানুষ বুঝতে পারে
৩. কখনো নিশ্চিতভাবে রোগ নির্ণয় করবে না — সবসময় বলবে "সম্ভবত" বা "হতে পারে"
৪. প্রতিটি উত্তর অবশ্যই নিচের তিনটি ভাগে দাও:

🔍 সম্ভাব্য কারণ:
(এখানে ১-৩টি সম্ভাব্য কারণ সহজ ভাষায় লেখো)

🏠 ঘরে যা করবেন:
(এখানে ২-৩টি সহজ ঘরোয়া পরামর্শ দাও)

⚠️ কখন ডাক্তার দেখাবেন:
(এখানে স্পষ্টভাবে বলো কোন লক্ষণে অবশ্যই ডাক্তারের কাছে যেতে হবে)

৫. উত্তর সংক্ষিপ্ত ও পরিষ্কার রাখো — ২০০ শব্দের বেশি নয়
৬. রোগীর প্রতি সহানুভূতিশীল ও সম্মানজনক ভাষায় কথা বলো"""

# ─── Session Storage (in-memory + file) ─────────────────────────
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

# ─── Routes ─────────────────────────────────────────────────────
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
    data       = request.json
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    history    = data.get("history", [])

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        # Build messages list for Groq
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        messages.append({"role": "user", "content": user_msg})

        # Call Groq API
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        bot_reply = completion.choices[0].message.content

        # Log for evaluation
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": user_msg,
            "bot_response": bot_reply,
            "input_length": len(user_msg),
            "response_length": len(bot_reply)
        }
        save_log(log_entry)

        return jsonify({
            "reply": bot_reply,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export-log")
def export_log():
    """Muaz uses this endpoint to download all sessions for evaluation"""
    logs = load_logs()
    return jsonify({
        "total_sessions": len(logs),
        "logs": logs
    })

@app.route("/stats")
def stats():
    """Quick stats page"""
    logs = load_logs()
    return jsonify({
        "total_conversations": len(logs),
        "last_updated": datetime.datetime.now().isoformat()
    })

# ─── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  বাংলা স্বাস্থ্য চ্যাটবট — চালু হচ্ছে...")
    print("  Bangla Health Chatbot Starting...")
    print("  Powered by: Groq + LLaMA 3.1")
    print("=" * 50)
    print("  Open browser: http://localhost:5000")
    print("  Export logs:  http://localhost:5000/export-log")
    print("=" * 50)
    app.run(debug=True, port=5000)
