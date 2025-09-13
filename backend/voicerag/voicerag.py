import os
import json
import asyncio
import queue
import threading
import time
import tempfile
import sounddevice as sd
import websocket
import simpleaudio as sa
import requests
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "chapters"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
def check_dependencies():
    required_vars = ["DEEPGRAM_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

check_dependencies()

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Audio queues and control
mic_queue = queue.Queue()
rag_queue = asyncio.Queue()
is_running = True
audio_buffer = []
silence_threshold = 0.01  # Adjust this value
min_speech_duration = 1.0  # Minimum 1 second of speech
silence_duration = 1.5  # 1.5 seconds of silence to trigger processing

# -----------------------------
# Audio Testing and Setup Functions
# -----------------------------

def test_microphone():
    """Test microphone input"""
    print("🎤 Testing microphone for 5 seconds...")
    print("Say something!")
    
    audio_levels = []
    
    def test_callback(indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        audio_levels.append(rms)
        if rms > 0.01:
            print(f"🔊 Audio detected: {rms:.4f}")
    
    try:
        with sd.InputStream(callback=test_callback, channels=1, samplerate=16000):
            time.sleep(5)
    except Exception as e:
        print(f"Microphone test error: {e}")
        return False
    
    max_level = max(audio_levels) if audio_levels else 0
    avg_level = sum(audio_levels) / len(audio_levels) if audio_levels else 0
    
    print(f"📊 Audio test results:")
    print(f"   Max level: {max_level:.4f}")
    print(f"   Avg level: {avg_level:.4f}")
    print(f"   Samples with audio > 0.01: {sum(1 for x in audio_levels if x > 0.01)}")
    
    if max_level < 0.001:
        print("⚠️  Very low audio levels detected. Check your microphone.")
    elif max_level < 0.01:
        print("⚠️  Low audio levels. You might need to speak louder or adjust microphone settings.")
    else:
        print("✅ Microphone working well!")
    
    return max_level > 0.001

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio callback status: {status}")
    
    # Calculate RMS (volume level)
    rms = np.sqrt(np.mean(indata**2))
    
    # Debug: Print audio level (comment out after testing)
    if rms > 0.001:  # Only print when there's some audio
        print(f"🎤 Audio level: {rms:.4f}")
    
    # Convert float32 to int16 PCM format for Deepgram
    audio_int16 = (indata * 32767).astype(np.int16)
    mic_queue.put((audio_int16.copy(), rms))

def start_mic_stream():
    try:
        # List available audio devices
        print("📻 Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"   {i}: {device['name']} (inputs: {device['max_input_channels']})")
        
        stream = sd.InputStream(
            channels=1,
            samplerate=16000,
            callback=audio_callback,
            blocksize=1024,
            dtype=np.float32
        )
        stream.start()
        print("🎤 Microphone stream started")
        return stream
    except Exception as e:
        print(f"Error starting microphone: {e}")
        return None

# -----------------------------
# 2️⃣ Streaming STT via Deepgram WS
# -----------------------------
class DeepgramSTT:
    def __init__(self):
        self.ws = None
        self.is_connected = False
        
    def start_stt_ws(self):
        ws_url = (
            "wss://api.deepgram.com/v1/listen?"
            "model=nova-2&"
            "language=en-US&"
            "encoding=linear16&"
            "sample_rate=16000&"
            "channels=1&"
            "punctuate=true&"
            "interim_results=true&"
            "endpointing=100&"
            "utterance_end_ms=1000&"
            "vad_events=true"
        )
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run WebSocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Start audio sending thread
        audio_thread = threading.Thread(target=self.send_audio_loop)
        audio_thread.daemon = True
        audio_thread.start()

    def on_open(self, ws):
        print("🔗 Connected to Deepgram")
        self.is_connected = True

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            
            # Debug: Print raw message
            print(f"🔍 Deepgram response: {json.dumps(data, indent=2)}")
            
            if "channel" in data:
                alternatives = data["channel"]["alternatives"]
                if alternatives:
                    transcript = alternatives[0]["transcript"].strip()
                    confidence = alternatives[0].get("confidence", 0)
                    is_final = data.get("is_final", True)
                    
                    print(f"📝 Transcript: '{transcript}' (confidence: {confidence:.2f}, final: {is_final})")
                    
                    # Process any non-empty transcript with reasonable confidence
                    if transcript and confidence > 0.5 and is_final:
                        print(f"✅ Processing: {transcript}")
                        # Put transcript in RAG queue
                        asyncio.run_coroutine_threadsafe(
                            rag_queue.put(transcript), 
                            asyncio.get_event_loop()
                        )
        except Exception as e:
            print(f"STT processing error: {e}")

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("🔌 Deepgram connection closed")
        self.is_connected = False

    def send_audio_loop(self):
        speech_detected = False
        silence_counter = 0
        speech_counter = 0
        
        while is_running:
            try:
                if self.is_connected and not mic_queue.empty():
                    audio_data, rms = mic_queue.get_nowait()
                    
                    # Send audio to Deepgram
                    if self.ws and self.ws.sock and self.ws.sock.connected:
                        self.ws.send(audio_data.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                    
                    # Speech detection logic
                    if rms > silence_threshold:
                        speech_counter += 1
                        silence_counter = 0
                        if not speech_detected:
                            speech_detected = True
                            print("🗣️ Speech detected!")
                    else:
                        silence_counter += 1
                        if speech_detected and silence_counter > (silence_duration * 16000 / 1024):
                            print(f"🔇 End of speech detected (spoke for {speech_counter * 1024 / 16000:.1f}s)")
                            speech_detected = False
                            speech_counter = 0
                            silence_counter = 0
                else:
                    time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Audio send error: {e}")
                time.sleep(0.1)

# -----------------------------
# 3️⃣ RAG pipeline
# -----------------------------
async def handle_rag():
    print("🤖 RAG handler started")
    while is_running:
        try:
            # Wait for user input with timeout
            user_text = await asyncio.wait_for(rag_queue.get(), timeout=1.0)
            print(f"📝 Processing: {user_text}")

            # Generate embedding for retrieval
            try:
                embedding_response = openai_client.embeddings.create(
                    input=user_text, 
                    model="text-embedding-3-small"
                )
                embedding = embedding_response.data[0].embedding
            except Exception as e:
                print(f"Embedding error: {e}")
                continue

            # Retrieve from Qdrant
            try:
                results = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=embedding,
                    limit=5,
                    score_threshold=0.5  # Only get relevant results
                )
                
                if results:
                    context_text = "\n".join([r.payload.get("text", "") for r in results])
                    print(f"📚 Found {len(results)} relevant documents")
                else:
                    context_text = "No relevant information found."
                    print("📚 No relevant documents found")
                    
            except Exception as e:
                print(f"Qdrant search error: {e}")
                context_text = "Error retrieving information."

            # Generate response with GPT
            try:
                prompt = f"""Answer the question using the provided context. If the context doesn't contain relevant information, say so politely.

Context:
{context_text}

Question: {user_text}

Answer (be concise and helpful):"""

                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                answer_text = response.choices[0].message.content.strip()
                print(f"💡 Answer: {answer_text}")

                # Convert to speech and play
                await play_tts_async(answer_text)
                
            except Exception as e:
                print(f"GPT generation error: {e}")
                await play_tts_async("Sorry, I encountered an error processing your request.")

        except asyncio.TimeoutError:
            # Continue the loop if no input received
            continue
        except Exception as e:
            print(f"RAG handler error: {e}")
            await asyncio.sleep(0.1)

# -----------------------------
# 4️⃣ Text-to-Speech
# -----------------------------
async def play_tts_async(text):
    """Convert text to speech and play it"""
    try:
        # Run TTS in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, play_tts_sync, text)
    except Exception as e:
        print(f"TTS async error: {e}")

def play_tts_sync(text):
    """Synchronous TTS function"""
    try:
        url = "https://api.deepgram.com/v1/speak?model=aura-luna-en"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}", 
            "Content-Type": "application/json"
        }
        payload = {"text": text}
        
        print("🔊 Generating speech...")
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(response.content)
                temp_filename = temp_file.name
            
            try:
                # Play the audio
                wave_obj = sa.WaveObject.from_wave_file(temp_filename)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                print("🔊 Speech playback completed")
            except Exception as e:
                print(f"Audio playback error: {e}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_filename)
                except:
                    pass
        else:
            print(f"TTS API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"TTS error: {e}")

# -----------------------------
# 5️⃣ Main application
# -----------------------------
async def main():
    global is_running
    
    print("🎙️ Starting real-time voice RAG pipeline...")
    print("Press Ctrl+C to stop.\n")

    # Test microphone first
    if not test_microphone():
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return

    # Test Qdrant connection
    try:
        collections = qdrant.get_collections()
        print(f"✅ Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
        
        # Check if our collection exists
        collection_exists = any(c.name == COLLECTION_NAME for c in collections.collections)
        if not collection_exists:
            print(f"⚠️  Warning: Collection '{COLLECTION_NAME}' not found in Qdrant")
        else:
            print(f"✅ Collection '{COLLECTION_NAME}' found")
            
    except Exception as e:
        print(f"❌ Qdrant connection error: {e}")
        return

    # Start microphone stream
    mic_stream = start_mic_stream()
    if not mic_stream:
        print("❌ Failed to start microphone")
        return

    try:
        # Initialize and start Deepgram STT
        stt = DeepgramSTT()
        stt.start_stt_ws()
        
        # Wait a moment for connection to establish
        await asyncio.sleep(2)
        
        # Start RAG handler
        rag_task = asyncio.create_task(handle_rag())
        
        print("✅ All systems ready!")
        print("🗣️  Start speaking... (I'll process your speech when you pause)")
        print("🔧 Debug mode is ON - you'll see audio levels and transcription details")
        
        # Keep the main loop running
        while is_running:
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        is_running = False
        if mic_stream:
            mic_stream.stop()
            mic_stream.close()
        print("👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")