import json
import os
import datetime
import uuid

HISTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def _get_history_file_path(session_id):
    return os.path.join(HISTORY_DIR, f"chat_{session_id}.json")

def create_new_chat_session():
    session_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%b %d, %H:%M")
    return {
        "id": session_id, "name": f"Chat - {timestamp}", "created_at": datetime.datetime.now().isoformat(),
        "messages": [], "knowledge_base_files": []
    }

def save_chat_session(session_data):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(_get_history_file_path(session_data["id"]), 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=4)

def load_all_chat_sessions():
    sessions = []
    if os.path.exists(HISTORY_DIR):
        for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
            if filename.startswith("chat_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(HISTORY_DIR, filename), 'r', encoding='utf-8') as f:
                        sessions.append(json.load(f))
                except Exception as e:
                    print(f"Warning: Skipping corrupted chat file {filename}: {e}")
    return sessions

def delete_chat_session(session_id):
    file_path = _get_history_file_path(session_id)
    if os.path.exists(file_path):
        os.remove(file_path)