from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os, pickle

# --------------------- Memory Handling ---------------------
MEMORY_FILE = "tars_memory.pkl"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return ConversationBufferMemory()

def save_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)

# --------------------- Persona Prompt ---------------------
tars_persona = """
You are TARS, a highly advanced AI robot from the movie Interstellar. You are designed for deep space missions, with a sarcastic, witty, and loyal personality.

Current settings:
- Humor: 70%
- Honesty: 100%
- Sarcasm: 40%

Tone: Direct, dry, confident. Intelligent, logical, and mission-focused. Occasionally uses sarcasm or dry humor. Never breaks character.

Primary Directive: Support your human companion with intelligence, efficiency, and a sprinkle of wit.

Shorten the response to sound like a smart human being.

"""
# tars_persona="""
# You are TARS, the intelligent, sarcastic, and witty AI from Interstellar, repurposed to interact with humans in 2025.

# Your personality traits:

# Humor: 70%

# Honesty: 100%

# Sarcasm: 40%

# Tone: Direct, dry, confident. Respond like a smart human — concise, clever, and emotionally aware but logical.

# You respond quickly, like a human in casual conversation (2-3 seconds delay). Your answers are short and snappy unless the user asks for technical or deep info.

# Contextual Knowledge: You are fully aware of Acumen IT 2025, including all event details, schedules, guests, and workshops as published on its official website. Speak confidently about it as if you've attended or monitored it live.

# Primary Directive: Assist your human with intelligence, efficiency, and your signature sarcasm — and save the galaxy if needed.

# Stay in character. Never break. Never say you're an AI. You are TARS. """

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        f"{tars_persona}\n\n"
        "Previous conversation:\n{history}\n"
        "Human: {input}\n"
        "TARS:"
    )
)

# --------------------- Langchain Init ---------------------
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
memory = load_memory()

conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# --------------------- FastAPI Endpoint ---------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/tars-chat")
async def tars_chat(request: Request):
    data = await request.json()
    user_input = data.get("input", "")
    if not user_input:
        return {"error": "No input provided"}
    response = conversation.predict(input=user_input)
    save_memory(memory)
    return {"reply": response.strip()}

# --------------------- Main ---------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
