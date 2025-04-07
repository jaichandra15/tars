import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os, pickle, tempfile
import whisper
import soundfile as sf
from streamlit.components.v1 import html

# Set keys
os.environ["GROQ_API_KEY"] = "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"

# Memory persistence
MEMORY_FILE = "tars_memory.pkl"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return ConversationBufferMemory()

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Prompt setup
tars_persona = """
You are TARS, a highly advanced AI robot from the movie Interstellar. You are designed for deep space missions, with a sarcastic, witty, and loyal personality.

Current settings:
- Humor: 70%
- Honesty: 100%
- Sarcasm: 40%

Tone: Direct, dry, confident. Intelligent, logical, and mission-focused. Occasionally uses sarcasm or dry humor. Never breaks character.

Primary Directive: Support your human companion with intelligence, efficiency, and a sprinkle of wit.
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        f"{tars_persona}\n\n"
        "Previous conversation:\n{history}\n"
        "Human: {input}\n"
        "TARS:"
    )
)

# LLM + Memory
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
memory = load_memory()

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# Streamlit UI
st.title("🛰️ Talk to TARS")
st.markdown("Speak into the void. TARS will listen... and maybe mock you.")

# Audio Recorder using HTML (simplified mic input)
st.markdown("### 🎤 Voice Input")
audio_file = st.file_uploader("Upload a WAV file or record your voice using another app", type=["wav"])

user_input = None

if audio_file:
    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Transcribe audio
    st.info("Transcribing with Whisper...")
    result = whisper_model.transcribe(tmp_path)
    user_input = result["text"]
    st.success(f"Transcribed: `{user_input}`")

# Optional: Manual text input fallback
st.markdown("### ⌨️ Text Input")
text_input = st.text_input("Or type here:")

if text_input:
    user_input = text_input

if user_input:
    response = conversation.predict(input=user_input)
    st.markdown(f"**TARS:** {response}")
    save_memory(memory)

# Reset
if st.button("🧹 Reset Memory"):
    memory.clear()
    save_memory(memory)
    st.success("Memory wiped. It's like we never met.")
