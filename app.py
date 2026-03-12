from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import chromadb
import ollama

# For now, use temporary storage for conversations. Later, I may adjust this
# approach to use persistent storage
conversations: dict[str, list[dict]] = {}

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="legal_contracts")

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:8000"
]

# CORS Middleware code to allow my frontend to communicate with my backend
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
    """
    Use Qwen3:32b to return an answer to the user's question

    user_prompt (Message): The user's question in the form of a Message object.
    session_id (string): Identifier used to locate the conversation history.

    Returns:
        dict: A dictionary containing the answer to the user's quesiton and the
              session id.
    """
    # Handle first message of a user session
    if not session_id or session_id not in conversations:
        session_id = str(uuid4())
        conversations[session_id] = []
    
    history = conversations[session_id]

    # Handle limited context by using a sliding window of 10 turns. Each turn 
    # (each call of get_answer) results in 2 new messages being appended to the
    # conversation, so 10 turns are stored in the conversation history at any
    # given time
    if len(history) >= 20:
        history[:] = history[2:]

    chat_text = ""

    for message in history:
        chat_text += message["content"]
    
    # Rewrite the user's question, taking into account the conversation history,
    # to ensure that it can be understood. This is essential for handling follow-up
    # questions which might be too vague to be understood without the context of the
    # conversation history
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

    # After testing, I decided that returning 10 results would produce the best
    # performance
    results = collection.query(
        query_texts=[rewritten_prompt["message"]["content"]],
        n_results=10
    )

    context = ""

    # Currently, the below code joins all titles followed by all chunks. In 
    # the future, I may revise this so that it is easier to determine which title
    # is associated with which chunk
    for i in range(len(results["documents"])):
        titles = " ".join([metadata["contract_title"] for metadata in results["metadatas"][i]])
        result = titles + " ".join(results["documents"][i])
        context += result

    # Add the retrieved clauses to the system prompt
    full_sys_prompt = f"""
                  {sys_prompt} The relevant contract clauses are as follows: {context}.
                  The prior history of this conversation is below, ending with the
                  user's current question. If there is no text provided other than a
                  question from the user, then this is the first question and there 
                  is no conversation history to reference.
                  """

    # The system prompt is not appended to the conversation history to avoid quickly
    # eating up the context window
    sys_message = {"role": "system", "content": full_sys_prompt}

    history.append({"role": user_prompt.role, "content": user_prompt.content})

    response = ollama.chat(
        model="qwen3:32b",
        messages=[sys_message] + history
    )

    answer = response["message"]["content"]

    history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "session_id": session_id}
