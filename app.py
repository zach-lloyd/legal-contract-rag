from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import chromadb
import ollama

conversations: dict[str, list[dict]] = {}

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="legal_contracts")

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

sys_prompt = """
             You are a helpful legal assistant. Provide a concise but 
             thorough answer to the user's latest question, using only
             the existing conversation history (if any) and the relevant clauses 
             from existing legal contracts provided below. If the answer is not 
             available. You should be honest about that. Do not hallucinate a 
             false answer. Rather, you should respond 'I don't have information about that.'
             """

@app.post("/chat/")
async def get_answer(user_prompt: Message, session_id: str = None):
    if not session_id or session_id not in conversations:
        session_id = str(uuid4())
        conversations[session_id] = []
    
    history = conversations[session_id]

    if len(history) >= 20:
        history[:] = history[2:]

    chat_text = ""

    for message in history:
        if message["role"] == "user" or message["role"] == "assistant":
            chat_text += message["content"]
    
    rewritten_prompt = ollama.chat(
        model="qwen3:32b",
        messages=[
            {
                "role": "system",
                "content": f"Conversation History: {chat_text}"
            },
            {
                "role": "system",
                "content": "Given this conversation history, rewrite the user's latest "
                           "question as a fully self-contained query in 256 words or less "
                           "that could be understood without any prior context."
            }, 
            {
                "role": "system",
                "content": f"User's Latest Question: {user_prompt.content}"
            }  
        ]
    )

    results = collection.query(
        query_texts=[rewritten_prompt["message"]["content"]],
        n_results=5
    )

    context = ""

    for i in range(len(results["documents"])):
        titles = " ".join([metadata["contract_title"] for metadata in results["metadatas"][i]])
        result = titles + " ".join(results["documents"][i])
        context += result

    full_sys_prompt = f"""
                  {sys_prompt} The relevant contract clauses are as follows: {context}.
                  The prior history of this conversation is below, ending with the
                  user's current question. If there is no text provided other than a
                  question from the user, then this is the first question and there 
                  is no conversation history to reference.
                  """

    sys_message = {"role": "system", "content": full_sys_prompt}

    history.append({"role": user_prompt.role, "content": user_prompt.content})

    response = ollama.chat(
        model="qwen3:32b",
        messages=[sys_message] + history
    )

    answer = response["message"]["content"]

    history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "session_id": session_id}
