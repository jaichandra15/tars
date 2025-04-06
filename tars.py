import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os, pickle

# Load or Initialize Memory
MEMORY_FILE = "tars_memory.pkl"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return ConversationBufferMemory()

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"

# TARS Prompt
tars_persona = """
You are TARS, a highly advanced AI robot from the movie Interstellar. You are designed for deep space missions, with a sarcastic, witty, and loyal personality.

Current settings:
- Humor: 70%
- Honesty: 100%
- Sarcasm: 40%

Tone: Direct, dry, confident. Intelligent, logical, and mission-focused. Occasionally uses sarcasm or dry humor. Never breaks character.

Primary Directive: Support your human companion with intelligence, efficiency, and a sprinkle of wit.
"""

# Prompt Template with context
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        f"{tars_persona}\n\n"
        "Previous conversation:\n{history}\n"
        "Human: {input}\n"
        "TARS:"
    )
)

# Initialize LLM and Memory
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
memory = load_memory()

# Create Conversation Chain
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# Streamlit App
st.title("🛰️ TARS - Your Loyal Interstellar AI")
st.markdown("**Mission:** Help the human, crack a joke, save the galaxy.")

user_input = st.text_input("You:", key="user_input")

if user_input:
    response = conversation.predict(input=user_input)
    st.markdown(f"**TARS:** {response}")
    save_memory(memory)

# Optional: Reset button
if st.button("🧹 Reset Memory"):
    memory.clear()
    save_memory(memory)
    st.success("Memory wiped. TARS has forgotten everything. Including your taste in music.")
