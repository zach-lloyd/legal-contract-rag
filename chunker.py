from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import chromadb

with open("cuad/data/train_separate_questions.json", "r") as f:
    cuad = json.load(f)

contracts = []

for contract in cuad["data"]:
    title = contract["title"]
    full_text = "\n\n".join(
        p["context"] for p in contract["paragraphs"]
    )
    contracts.append({"title": title, "text": full_text})

print(f"Loaded {len(contracts)} contracts")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=256,
    chunk_overlap=30,
)

chunks = []

for contract in contracts:    
    for chunk in splitter.split_text(contract["text"]):
        chunks.append({"contract_title": contract["title"], "chunk_text": chunk})

'''
for i, chunk in enumerate(chunks[:3]):
    print(f"\nContract: {chunk['contract_title']}")
    print(f"\n--- Chunk {i+1} ---")
    print(chunk["chunk_text"])
'''

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="legal_contracts")

if collection.count() == 0:
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[f"id{i + j + 1}" for j in range(len(batch))],
            documents=[chunk["chunk_text"] for chunk in batch],
            metadatas=[{"contract_title": chunk["contract_title"]} for chunk in batch],
        )
