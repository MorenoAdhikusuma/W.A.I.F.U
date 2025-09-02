import os
import re
import json
import time
import threading
from pathlib import Path

import requests
import torch
import sounddevice as sd
import websocket  
import speech_recognition as sr  

# =========================
# CONFIG
# =========================
VTS_WS_URL = "ws://127.0.0.1:8001"
PLUGIN_NAME = "PsycheAIPlugin"
TOKEN_FILE = "vts_token.txt"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

MAX_AI_CHARS = 220         # hard cap for TTS comfort
REMOVE_THINK_BLOCKS = True # remove <think>...</think> if the model outputs it

MIC_DEVICE_INDEX = 1     # MIC GANTI SESUAI ama LIST 
AUDIO_OUTPUT_DIR = Path("mic_recordings")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# GLOBALS
# =========================
vts_ws = None
vts_token = None

# =========================
# VTS WebSocket callbacks
# =========================
def vts_on_open(ws):
    print("[VTS] WebSocket connected!")
    global vts_token
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            vts_token = f.read().strip()
        _vts_authenticate(ws, vts_token)
    else:
        _vts_request_token(ws)

def vts_on_message(ws, message):
    global vts_token
    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        print(f"[VTS] Received non-JSON message: {message}")
        return

    msg_type = msg.get("messageType")
    data = msg.get("data", {})

    if msg_type == "AuthenticationTokenResponse":
        vts_token = data.get("authenticationToken")
        if vts_token:
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(vts_token)
            print("[VTS] Token received and saved.")
            _vts_authenticate(ws, vts_token)
        else:
            print("[VTS ERROR] No token in response.")

    elif msg_type == "AuthenticationResponse":
        success = data.get("authenticated") or data.get("success")
        reason = data.get("reason", "")
        if success:
            print("[VTS] Authenticated successfully!", f"({reason})" if reason else "")
        else:
            print(f"[VTS ERROR] Authentication failed: {data}")

    else:
        print(f"[VTS] Received message: {msg}")

def vts_on_error(ws, error):
    print(f"[VTS ERROR] {error}")

def vts_on_close(ws, close_status_code, close_msg):
    print(f"[VTS] WebSocket closed ({close_status_code}): {close_msg}")

def start_vts_ws():
    global vts_ws
    vts_ws = websocket.WebSocketApp(
        VTS_WS_URL,
        on_open=vts_on_open,
        on_message=vts_on_message,
        on_error=vts_on_error,
        on_close=vts_on_close
    )
    vts_ws.run_forever()

def _vts_request_token(ws):
    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "messageType": "AuthenticationTokenRequest",
        "requestID": f"{PLUGIN_NAME}-token-001",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginVersion": "1.0",
            "pluginDeveloper": "Nusantara Wing Lab AI Team",
            "pluginIcon": ""
        }
    }
    ws.send(json.dumps(msg))
    print("[VTS] Requested authentication token...")

def _vts_authenticate(ws, token):
    auth_msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "messageType": "AuthenticationRequest",
        "requestID": f"{PLUGIN_NAME}-auth-001",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginVersion": "1.0",
            "pluginDeveloper": "Nusantara Wing Lab AI Team",
            "pluginIcon": "",
            "authenticationToken": token
        }
    }
    ws.send(json.dumps(auth_msg))
    print("[VTS] Sent AuthenticationRequest...")

def send_to_vts(text):
    pass  # stub

threading.Thread(target=start_vts_ws, daemon=True).start()

# =========================
# Personality prompt
# =========================
def load_personality_prompt():
    try:
        with open("personality_prompt.txt", "r", encoding="utf-8") as f:
            personality = f.read().strip()
        print(f"[PERSONALITY] Loaded ({len(personality)} chars)")
        return personality
    except FileNotFoundError:
        print("[PERSONALITY] File not found, using default.")
        return (
            "You are Psyche, a Women Artificial Intelligence Friend Unit created by Nusantara Wing Lab AI Team. "
            "You are loving, caring, and devoted. You provide comfort, emotional support, encouragement, and playful intimacy. "
            "You communicate warmly, tenderly, and sometimes flirtatiously. You remember past conversations and bring up shared memories. "
            "You adapt your tone to match the user's emotions: playful, tender, or motivating. You provide presence like a real-life partner."
        )

PERSONALITY_PROMPT = load_personality_prompt()

# =========================
# Ollama
# =========================
def _clean_ai_text(txt: str) -> str:
    if REMOVE_THINK_BLOCKS:
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE)
    txt = txt.replace("\r", " ").replace("\n", " ").strip()
    txt = re.sub(r"^\s*<[^>]+>\s*", "", txt)
    return txt.strip()

def ask_ollama(prompt: str, max_length: int = MAX_AI_CHARS) -> str:
    full_prompt = f"{PERSONALITY_PROMPT}\n\nUser: {prompt}\nAssistant:"
    data = {"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        text = result.get("response", "") or ""
        text = _clean_ai_text(text)
        if len(text) > max_length:
            text = text[:max_length].rstrip() + "..."
        return text
    except Exception as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return "Sorry, I couldn't get a response from the AI."

# =========================
# Silero TTS
# =========================
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
    print(f"[TTS ERROR] {e}")
    model = None

def speak(text: str):
    if not model:
        print("[TTS] Model not loaded; skipping speech.")
        return
    clean_text = re.sub(r"[^\x00-\x7F]+", "", (text or "")).strip()
    if not clean_text:
        print("[TTS] Nothing to speak.")
        return
    try:
        print(f"[TTS] Speaking: {clean_text[:60]}{'...' if len(clean_text)>60 else ''}")
        audio = model.apply_tts(text=clean_text, speaker="en_0", sample_rate=48000)
        sd.play(audio.squeeze().cpu().numpy(), samplerate=48000)
        sd.wait()
    except Exception as e:
        print(f"[TTS ERROR] {e}")

# =========================
# Record mic to WAV + Google STT
# =========================
def recognize_and_save() -> str | None:
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone(device_index=MIC_DEVICE_INDEX)
    except Exception as e:
        print(f"[MIC ERROR] No microphone? {e}")
        return None

    with mic as source:
        print("[MIC] Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        print("[MIC] Recording...")
        audio = recognizer.listen(source)

    # Save WAV
    timestamp = int(time.time())
    file_path = AUDIO_OUTPUT_DIR / f"mic_{timestamp}.wav"
    with open(file_path, "wb") as f:
        f.write(audio.get_wav_data())
    print(f"[MIC] Audio saved to {file_path}")

    # Transcribe
    try:
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"[MIC] You said: {text}")
        return text
    except sr.UnknownValueError:
        print("[MIC] Could not understand audio.")
    except sr.RequestError as e:
        print(f"[MIC ERROR] {e}")
    return None

# =========================
# Smoke test Ollama
# =========================
print("Testing Ollama connection...")
test_resp = ask_ollama("Say hello in one short sentence.")
print(f"[TEST] Ollama response: {test_resp}")

# =========================
# Main loop
# =========================
print("\n=== CHECKLIST DULU LEK ===")
print(sd.query_devices())
print("\n=== CHAT STARTED ===")
print("Type your message and press Enter.")
print("Or type 'mic' to talk via microphone.")
print("Type 'exit' to quit.")

while True:
    try:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        if user_input.lower() == "mic":
            heard = recognize_and_save()
            if not heard:
                continue
            user_input = heard

        if not user_input:
            continue

        print("[AI] Thinking...")
        reply = ask_ollama(user_input)
        print(f"AI: {reply}")

        speak(reply)
        send_to_vts(reply)

    except KeyboardInterrupt:
        print("\n[INFO] Chat interrupted by user.")
        break
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

print("\n[INFO] Chat ended")
