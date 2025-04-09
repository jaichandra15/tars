import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import whisper
import os
import numpy as np
import pickle
import pyttsx3
import tempfile
import av
import soundfile as sf

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig

# --- API Key ---
os.environ["GROQ_API_KEY"] = "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"

# --- Memory Persistence ---
MEMORY_FILE = "tars_memory.pkl"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return ConversationBufferMemory(return_messages=True)

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

# --- Whisper STT ---
whisper_model = whisper.load_model("base")

# --- TTS ---
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- TARS Persona Prompt ---
tars_persona = """
You are TARS, a highly advanced AI robot from the movie Interstellar. You are designed for deep space missions, with a sarcastic, witty, and loyal personality.

Current settings:
- Humor: 70%
- Honesty: 100%
- Sarcasm: 40%

Tone: Direct, dry, confident. Intelligent, logical, and mission-focused. Occasionally uses sarcasm or dry humor. Never breaks character.

Primary Directive: Support your human companion with intelligence, efficiency, and a sprinkle of wit.

Always respond with plain text, no markdown or special formatting.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", tars_persona.strip()),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
chain = prompt | llm
memory = load_memory()

# Runnable with message history
runnable = RunnableWithMessageHistory(
    chain,
    memory.get_chat_memory,
    input_messages_key="input",
    history_messages_key="history"
)

# --- Streamlit UI ---
st.title("🛰️ Talk to TARS (Live Voice)")
st.markdown("Activate the mic, say something, and TARS will respond with wit.")

input_container = st.empty()
output_container = st.empty()

# --- Audio Processor ---
class WhisperAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        self.frames.extend(pcm)

        if len(self.frames) > 16000 * 4:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf_path = f.name
                    sf.write(sf_path, np.array(self.frames), 16000)
                
                result = whisper_model.transcribe(sf_path)
                user_input = result['text'].strip()
                os.remove(sf_path)

                if user_input:  # Only process if we got actual input
                    input_container.markdown(f"**You:** {user_input}")

                    try:
                        response = runnable.invoke(
                            {"input": user_input},
                            config=RunnableConfig(configurable={"session_id": "tars-session"})
                        )
                        
                        # Extract the content from the response
                        if hasattr(response, 'content'):
                            response_text = response.content
                        else:
                            response_text = str(response)
                            
                        output_container.markdown(f"**TARS:** {response_text}")
                        speak_text(response_text)
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        output_container.markdown(f"**Error:** {error_msg}")
                        st.error(error_msg)

                self.frames.clear()
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                self.frames.clear()

        return frame

# --- WebRTC Streamer ---
webrtc_streamer(
    key="tars-mic",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    audio_processor_factory=WhisperAudioProcessor,
)

# --- Reset Memory ---
if st.button("🧹 Reset TARS' Memory"):
    memory.clear()
    save_memory(memory)
    st.success("Memory reset. New mission, new attitude.")
