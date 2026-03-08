import chromadb
import ollama

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="legal_contracts")
NUM_RESULTS = 5

messages = []
num_turns = 0

while True:
    if num_turns >= 10: messages = messages[3:]

    prompt = input("Enter your prompt:")

    convo_history = ""

    for message in messages:
        if message["role"] == "user" or message["role"] == "assistant":
            convo_history += message["content"]

    rewritten_prompt = ollama.chat(
        model="qwen3:32b",
        messages=[
            {
                "role": "system",
                "content": f"Conversation History: {convo_history}"
            },
            {
                "role": "system",
                "content": "Given this conversation history, rewrite the user's latest "
                           "question as a fully self-contained query in 256 words or less "
                           "that could be understood without any prior context."
            }, 
            {
                "role": "system",
                "content": f"User's Latest Question: {prompt}"
            }  
        ]
    )

    results = collection.query(
        query_texts=[rewritten_prompt["message"]["content"]],
        n_results=NUM_RESULTS
    )

    context = ""

    for i in range(len(results["documents"])):
        titles = " ".join([metadata["contract_title"] for metadata in results["metadatas"][i]])
        result = titles + " ".join(results["documents"][i])
        context += result
    
    messages.extend([
            {"role": "system", 
            "content": "You are a helpful legal assistant. Provide a concise but " +
                       "thorough answer to the user's question, using only " +
                       "the relevant clauses from existing legal contracts provided below. " +
                       "If the answer is not available. You should be honest about that. Do "
                       "not hallucinate a false answer. Rather, you should respond " +
                       "'I don't have information about that.' The relevant contract clauses" +
                      f"are as follows: {context}"
            },
            {"role": "user", "content": prompt}
    ])

    response = ollama.chat(
        model="qwen3:32b",
        messages=messages
    )

    answer = response["message"]["content"]

    print(answer)

    messages.append({"role": "assistant", "content": answer})

    num_turns += 1
