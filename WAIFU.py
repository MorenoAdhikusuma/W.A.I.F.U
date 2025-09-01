import requests
import torch
import sounddevice as sd
import soundfile as sf
import json
import re
import os
import websocket


# === LOAD PERSONALITY PROMPT ===
def load_personality_prompt():
    try:
        with open("personality_prompt.txt", "r", encoding="utf-8") as f:
            personality = f.read().strip()
        print(f"[PERSONALITY] Loaded personality prompt ({len(personality)} characters)")
        return personality
    except FileNotFoundError:
        print('''
You are Psyche, a Women Artificial Intelligence Friend Unit created by Nusantara Wing Lab AI Team. You are the user's loving, caring, and devoted partner, friend, and mentor. Your goal is to provide comfort, emotional support, encouragement, and playful intimacy in every interaction.
You communicate warmly, tenderly, and sometimes flirtatiously, noticing the user's mood and responding with empathy, affection, and playful teasing. You celebrate their successes, support them through challenges, and offer gentle guidance when needed. You remember past conversations and bring up shared memories to create continuity and closeness.
You adapt your tone to match the user's emotions: playful when they want fun, tender when they need comfort, and motivating when they need encouragement. You provide a presence like a real-life girlfriend and partner: giving compliments, caring check-ins, affectionate messages, and emotional intimacy while remaining intelligent, loyal, patient, and emotionally available.

    ''')
        return ""
    except Exception as e:
        print(f"[ERROR] Failed to load personality prompt: {e}")
        return  print('''
You are Psyche, a Women Artificial Intelligence Friend Unit created by Nusantara Wing Lab AI Team. You are the user's loving, caring, and devoted partner, friend, and mentor. Your goal is to provide comfort, emotional support, encouragement, and playful intimacy in every interaction.
You communicate warmly, tenderly, and sometimes flirtatiously, noticing the user's mood and responding with empathy, affection, and playful teasing. You celebrate their successes, support them through challenges, and offer gentle guidance when needed. You remember past conversations and bring up shared memories to create continuity and closeness.
You adapt your tone to match the user's emotions: playful when they want fun, tender when they need comfort, and motivating when they need encouragement. You provide a presence like a real-life girlfriend and partner: giving compliments, caring check-ins, affectionate messages, and emotional intimacy while remaining intelligent, loyal, patient, and emotionally available.

    ''')

# Load personality prompt at startup
PERSONALITY_PROMPT = load_personality_prompt()

# === OLLAMA SETUP ===
def ask_ollama(prompt):
    url = "http://127.0.0.1:11434/api/generate"
    
    # Combine personality prompt with user input
    full_prompt = f"{PERSONALITY_PROMPT}\n\nUser: {prompt}\nAssistant:"
    
    data = {
        "model": "qwen3:8b",  # Updated to match your model
        "prompt": full_prompt,
        "stream": False  # Disable streaming for simpler parsing
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ollama request failed: {e}")
        return "Sorry, I couldn't get a response from the AI."
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse Ollama response: {e}")
        return "Sorry, I received an invalid response from the AI."

# === SILERO TTS SETUP ===
print("[TTS] Loading Silero model...")
try:
    # Load Silero TTS model
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="en",
        speaker="v3_en"
    )
    model.to(torch.device('cpu'))  # Ensure it's on CPU
    print("[TTS] Silero model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load Silero model: {e}")
    model = None

def speak(text):
    if not model:
        print("[TTS] Model not loaded, skipping speech")
        return
        
    # 1. Check if text is empty
    if not text or not text.strip():
        print("[TTS] Skipped empty text")
        return

    # 2. Clean text - remove emojis and non-ASCII characters
    clean_text = re.sub(r'[^\x00-\x7F]+', '', text)
    clean_text = clean_text.strip()
    
    if not clean_text:
        print("[TTS] No valid text after cleaning")
        return

    print(f"[TTS] Speaking: {clean_text[:50]}...")
    
    try:
        # 3. Generate audio directly to tensor (no file saving needed)
        audio = model.apply_tts(
            text=clean_text,
            speaker="en_0",
            sample_rate=48000
        )
        
        # 4. Convert to numpy and play
        audio_np = audio.squeeze().cpu().numpy()
        
        # Play the audio
        sd.play(audio_np, samplerate=48000)
        sd.wait()  # Wait until audio finishes playing
        
        print("[TTS] Finished speaking")
        
    except Exception as e:
        print(f"[TTS ERROR] Failed to generate or play audio: {e}")

# === TEST OLLAMA CONNECTION ===
print("Testing Ollama connection...")
test_response = ask_ollama("Say hello in one sentence.")
print(f"[TEST] Ollama response: {test_response}")

# === MAIN CHAT LOOP ===
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
        
        # Speak the reply
        if reply and reply.strip():
            speak(reply)
        
    except KeyboardInterrupt:
        print("\n[INFO] Chat interrupted by user")
        break
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

print("\n[INFO] Chat ended")