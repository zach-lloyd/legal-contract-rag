from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import chromadb
import ollama

# For now, use temporary storage for conversations. Later, I may adjust this
# approach to use persistent storage
conversations: dict[str, list[dict]] = {}
clauses: dict[str, list[str]] = {}

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
             You are a helpful legal assistant. Provide a concise but thorough 
             answer to the user's latest question, using the relevant clauses 
             from existing legal contracts provided below. The prior history of this
             conversation, if any, is also provided below. You may use it as additional
             context when providing your answer, but your answer should be primarily
             based on the relevant contract clauses that have been provided. If
             the prior answer references a specific contract, you should bias
             towards focusing on that contract in referencing the user's current
             question, unless their question makes it clear that they want you to 
             refer to other contracts. If the answer is not available. You should 
             be honest about that. Do not hallucinate a false answer. Rather, you 
             should respond 'I don't have information about that.'
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
        clauses[session_id] = []
    
    history = conversations[session_id]

    # Handle limited context by using a sliding window of 10 turns. Each turn 
    # (each call of get_answer) results in 2 new messages being appended to the
    # conversation, so 10 turns are stored in the conversation history at any
    # given time
    if len(history) >= 20:
        history[:] = history[2:]

    prompt = ""
    # If there is a conversation history, rewrite the user's question, taking 
    # into account the history, to ensure that it can be understood. This is 
    # essential for handling follow-up questions which might be too vague to 
    # be understood without the context of the conversation history
    if len(history) > 0:
        chat_text = ""

        for message in history:
            chat_text += f"{message['role']}: "
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
                    # Limit to 256 words to avoid eating up too much of the context window
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

        prompt = rewritten_prompt["message"]["content"]
    else:
        prompt = user_prompt.content

    # After testing, I decided that returning 10 results would produce the best
    # performance
    results = collection.query(
        query_texts=[prompt],
        n_results=10
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    clauses_list = clauses[session_id]

    for meta, chunk in zip(metadatas, chunks):
       print(meta['contract_title'])
       title_and_excerpt = f"Contract Title: {meta['contract_title']}\nContract Excerpt: {chunk}\n\n"

       if title_and_excerpt not in clauses_list:
           clauses_list.append(title_and_excerpt)
    
    if len(clauses_list) > 30:
        clauses_list[:] = clauses_list[-30:]
           
    context = " ".join(clauses_list)

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
