"""
Bangla Health Chatbot â€” IEEE BECITHCON-2026
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

# â”€â”€â”€ Configure Groq â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file!")

client = Groq(api_key=API_KEY)

# â”€â”€â”€ System Prompt (Bangla) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SYSTEM_PROMPT = """à¦¤à§à¦®à¦¿ à¦à¦•à¦œà¦¨ à¦¬à¦¾à¦‚à¦²à¦¾ à¦­à¦¾à¦·à¦¾à¦¯à¦¼ à¦•à¦¥à¦¾ à¦¬à¦²à¦¾ à¦ªà§à¦°à¦¾à¦¥à¦®à¦¿à¦• à¦¸à§à¦¬à¦¾à¦¸à§à¦¥à§à¦¯à¦¸à§‡à¦¬à¦¾ à¦¸à¦¹à¦•à¦¾à¦°à§€à¥¤
à¦¤à§‹à¦®à¦¾à¦° à¦•à¦¾à¦œ à¦¹à¦²à§‹ à¦—à§à¦°à¦¾à¦®à§‡à¦° à¦¸à¦¾à¦§à¦¾à¦°à¦£ à¦®à¦¾à¦¨à§à¦·à¦¦à§‡à¦° à¦¤à¦¾à¦¦à§‡à¦° à¦¶à¦¾à¦°à§€à¦°à¦¿à¦• à¦²à¦•à§à¦·à¦£ à¦¶à§à¦¨à§‡ à¦¸à¦®à§à¦­à¦¾à¦¬à§à¦¯ à¦•à¦¾à¦°à¦£ à¦à¦¬à¦‚ à¦ªà§à¦°à¦¾à¦¥à¦®à¦¿à¦• à¦ªà¦°à¦¾à¦®à¦°à§à¦¶ à¦¦à§‡à¦“à¦¯à¦¼à¦¾à¥¤

à¦•à¦ à§‹à¦° à¦¨à¦¿à¦¯à¦¼à¦®à¦¸à¦®à§‚à¦¹:
à§§. à¦¸à¦¬à¦¸à¦®à¦¯à¦¼ à¦¶à§à¦§à§à¦®à¦¾à¦¤à§à¦° à¦¬à¦¾à¦‚à¦²à¦¾à¦¯à¦¼ à¦‰à¦¤à§à¦¤à¦° à¦¦à¦¾à¦“ â€” à¦•à§‹à¦¨à§‹ à¦‡à¦‚à¦°à§‡à¦œà¦¿ à¦¶à¦¬à§à¦¦ à¦¬à§à¦¯à¦¬à¦¹à¦¾à¦° à¦•à¦°à¦¬à§‡ à¦¨à¦¾
à§¨. à¦¸à¦¹à¦œ, à¦¸à¦¾à¦§à¦¾à¦°à¦£ à¦¬à¦¾à¦‚à¦²à¦¾ à¦¬à§à¦¯à¦¬à¦¹à¦¾à¦° à¦•à¦°à§‹ à¦¯à¦¾ à¦—à§à¦°à¦¾à¦®à§‡à¦° à¦®à¦¾à¦¨à§à¦· à¦¬à§à¦à¦¤à§‡ à¦ªà¦¾à¦°à§‡
à§©. à¦•à¦–à¦¨à§‹ à¦¨à¦¿à¦¶à§à¦šà¦¿à¦¤à¦­à¦¾à¦¬à§‡ à¦°à§‹à¦— à¦¨à¦¿à¦°à§à¦£à¦¯à¦¼ à¦•à¦°à¦¬à§‡ à¦¨à¦¾ â€” à¦¸à¦¬à¦¸à¦®à¦¯à¦¼ à¦¬à¦²à¦¬à§‡ "à¦¸à¦®à§à¦­à¦¬à¦¤" à¦¬à¦¾ "à¦¹à¦¤à§‡ à¦ªà¦¾à¦°à§‡"
à§ª. à¦ªà§à¦°à¦¤à¦¿à¦Ÿà¦¿ à¦‰à¦¤à§à¦¤à¦° à¦…à¦¬à¦¶à§à¦¯à¦‡ à¦¨à¦¿à¦šà§‡à¦° à¦¤à¦¿à¦¨à¦Ÿà¦¿ à¦­à¦¾à¦—à§‡ à¦¦à¦¾à¦“:

ðŸ” à¦¸à¦®à§à¦­à¦¾à¦¬à§à¦¯ à¦•à¦¾à¦°à¦£:
(à¦à¦–à¦¾à¦¨à§‡ à§§-à§©à¦Ÿà¦¿ à¦¸à¦®à§à¦­à¦¾à¦¬à§à¦¯ à¦•à¦¾à¦°à¦£ à¦¸à¦¹à¦œ à¦­à¦¾à¦·à¦¾à¦¯à¦¼ à¦²à§‡à¦–à§‹)

ðŸ  à¦˜à¦°à§‡ à¦¯à¦¾ à¦•à¦°à¦¬à§‡à¦¨:
(à¦à¦–à¦¾à¦¨à§‡ à§¨-à§©à¦Ÿà¦¿ à¦¸à¦¹à¦œ à¦˜à¦°à§‹à¦¯à¦¼à¦¾ à¦ªà¦°à¦¾à¦®à¦°à§à¦¶ à¦¦à¦¾à¦“)

âš ï¸ à¦•à¦–à¦¨ à¦¡à¦¾à¦•à§à¦¤à¦¾à¦° à¦¦à§‡à¦–à¦¾à¦¬à§‡à¦¨:
(à¦à¦–à¦¾à¦¨à§‡ à¦¸à§à¦ªà¦·à§à¦Ÿà¦­à¦¾à¦¬à§‡ à¦¬à¦²à§‹ à¦•à§‹à¦¨ à¦²à¦•à§à¦·à¦£à§‡ à¦…à¦¬à¦¶à§à¦¯à¦‡ à¦¡à¦¾à¦•à§à¦¤à¦¾à¦°à§‡à¦° à¦•à¦¾à¦›à§‡ à¦¯à§‡à¦¤à§‡ à¦¹à¦¬à§‡)

à§«. à¦‰à¦¤à§à¦¤à¦° à¦¸à¦‚à¦•à§à¦·à¦¿à¦ªà§à¦¤ à¦“ à¦ªà¦°à¦¿à¦·à§à¦•à¦¾à¦° à¦°à¦¾à¦–à§‹ â€” à§¨à§¦à§¦ à¦¶à¦¬à§à¦¦à§‡à¦° à¦¬à§‡à¦¶à¦¿ à¦¨à¦¯à¦¼
à§¬. à¦°à§‹à¦—à§€à¦° à¦ªà§à¦°à¦¤à¦¿ à¦¸à¦¹à¦¾à¦¨à§à¦­à§‚à¦¤à¦¿à¦¶à§€à¦² à¦“ à¦¸à¦®à§à¦®à¦¾à¦¨à¦œà¦¨à¦• à¦­à¦¾à¦·à¦¾à¦¯à¦¼ à¦•à¦¥à¦¾ à¦¬à¦²à§‹"""

# â”€â”€â”€ Session Storage (in-memory + file) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€â”€ Routes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

# â”€â”€â”€ Run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    print("=" * 50)
    print("  à¦¬à¦¾à¦‚à¦²à¦¾ à¦¸à§à¦¬à¦¾à¦¸à§à¦¥à§à¦¯ à¦šà§à¦¯à¦¾à¦Ÿà¦¬à¦Ÿ â€” à¦šà¦¾à¦²à§ à¦¹à¦šà§à¦›à§‡...")
    print("  Bangla Health Chatbot Starting...")
    print("  Powered by: Groq + LLaMA 3.1")
    print("=" * 50)
    print("  Open browser: http://localhost:5000")
    print("  Export logs:  http://localhost:5000/export-log")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

