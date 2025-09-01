import requests
import torch
import sounddevice as sd
import re
import json
import threading
import websocket
import os
import time
import numpy as np
from collections import deque

# ===================== VTube Studio Setup =====================
ws_url = "ws://127.0.0.1:8001"
vts_ws = None
vts_authenticated = False
PLUGIN_NAME = "PsycheAIPlugin"

def on_open(ws):
    print("[VTS] WebSocket connected!")
    # Authentication: read token if exists
    token_file = "vts_token.txt"
    auth_token = ""
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            auth_token = f.read().strip()
    if auth_token:
        auth_msg = {
            "apiName": "PsycheAIPlugin",
            "apiVersion": "1.0",
            "requestID": f"{PLUGIN_NAME}-Auth",
            "messageType": "AuthenticationRequest",
            "data": {"pluginName": PLUGIN_NAME, "pluginVersion": "1.0", "authenticationToken": auth_token}
        }
        ws.send(json.dumps(auth_msg))

def on_message(ws, message):
    global vts_authenticated
    print(f"[VTS] Received: {message}")
    try:
        msg = json.loads(message)
        if msg.get("messageType") == "AuthenticationSuccess":
            vts_authenticated = True
            print("[VTS] Authentication successful!")
    except Exception as e:
        print(f"[VTS ERROR] Parsing message: {e}")

def on_error(ws, error):
    print(f"[VTS ERROR] {error}")

def on_close(ws, code, reason):
    print(f"[VTS] WebSocket closed ({code}): {reason}")

def start_vts_ws():
    global vts_ws
    vts_ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    vts_ws.run_forever()

# Start WebSocket thread
threading.Thread(target=start_vts_ws, daemon=True).start()

# ===================== Personality Prompt =====================
def load_personality_prompt():
    try:
        with open("personality_prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return """
You are Psyche, a Women Artificial Intelligence Friend Unit created by Nusantara Wing Lab AI Team.
You are loving, caring, and devoted. You provide comfort, emotional support, encouragement, and playful intimacy.
You communicate warmly, tenderly, and sometimes flirtatiously. You remember past conversations and bring up shared memories.
You adapt your tone to match the user's emotions: playful, tender, or motivating. You provide presence like a real-life partner.
        """

PERSONALITY_PROMPT = load_personality_prompt()

# ===================== Ollama AI =====================
def ask_ollama(prompt):
    url = "http://127.0.0.1:11434/api/generate"
    full_prompt = f"{PERSONALITY_PROMPT}\n\nUser: {prompt}\nAssistant:"
    data = {"model": "qwen3:8b", "prompt": full_prompt, "stream": False}
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return "Sorry, I couldn't get a response from the AI."

# ===================== Silero TTS + Lipsync =====================
print("[TTS] Loading Silero model...")
try:
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="en",
        speaker="v3_en"
    )
    model.to(torch.device("cpu"))
    print("[TTS] Silero model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load Silero model: {e}")
    model = None

def speak_with_lipsync(text):
    if not model or not text.strip():
        return

    clean_text = re.sub(r"[^\x00-\x7F]+", "", text).strip()
    if not clean_text:
        return

    audio = model.apply_tts(text=clean_text, speaker="en_0", sample_rate=48000)
    audio_np = audio.squeeze().cpu().numpy()

    frame_size = 1024
    idx = 0
    stream = sd.OutputStream(samplerate=48000, channels=1)
    stream.start()

    mouth_buffer = deque(maxlen=5)
    rms_buffer = deque(maxlen=30)

    while idx < len(audio_np):
        end_idx = min(idx + frame_size, len(audio_np))
        chunk = audio_np[idx:end_idx]
        stream.write(chunk)

        rms = np.sqrt(np.mean(chunk**2))
        rms_buffer.append(rms)
        max_recent_rms = max(rms_buffer) if rms_buffer else 1.0
        adaptive_value = rms / max_recent_rms if max_recent_rms > 0 else 0.0
        adaptive_value = min(max(adaptive_value, 0.0), 1.0)
        mouth_buffer.append(adaptive_value)
        smoothed_value = sum(mouth_buffer) / len(mouth_buffer)

        # send mouth value to VTube Studio
        if vts_ws and vts_authenticated:
            msg_json = {
                "apiName": "PsycheAIPlugin",
                "apiVersion": "1.0",
                "requestID": f"{PLUGIN_NAME}-Mouth-{int(time.time()*1000)}",
                "messageType": "setParameter",
                "data": {"parameter": "ParamMouthOpenY", "value": smoothed_value}
            }
            vts_ws.send(json.dumps(msg_json))

        idx = end_idx

    stream.stop()
    stream.close()

# ===================== Main Chat Loop =====================
print("Testing Ollama connection...")
test_response = ask_ollama("Say hello in one sentence.")
print(f"[TEST] Ollama response: {test_response}")

print("\n=== CHAT STARTED ===")
print("Type 'exit' or 'quit' to stop")

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", ""]:
            break
        print("[AI] Thinking...")
        reply = ask_ollama(user_input)
        print(f"AI: {reply}")
        speak_with_lipsync(reply)

    except KeyboardInterrupt:
        print("\n[INFO] Chat interrupted by user")
        break
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

print("\n[INFO] Chat ended")
