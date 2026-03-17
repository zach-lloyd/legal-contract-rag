# Legal Contract RAG Agent
 
A retrieval-augmented generation (RAG) agent that answers natural language questions about legal contracts. The system retrieves relevant contract clauses from a chromadb collection and uses a large language model to generate grounded, citation-aware answers. It supports multi-turn conversations with follow-up questions, cross-contract comparisons, and honest "I don't know" responses when the answer isn't available. This is a full-stack RAG with a simple front-end interface.
 
Built on the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad), a corpus of 510 commercially negotiated legal contracts annotated by legal experts across 41 clause categories.
 
## Architecture
 
The system is a FastAPI backend with the following pipeline:
 
1. **Query rewriting** — When a conversation has prior history, the user's question is rewritten into a self-contained query using the LLM. This ensures follow-up questions like "Are there any exceptions to those?" are expanded with the necessary context (e.g., which contract and clause type the user is referring to) before retrieval.
 
2. **Retrieval** — The query is embedded and compared against contract chunks stored in ChromaDB. The top 10 results are returned, each paired with the title of the contract from which it was extracted.
 
3. **Clause caching** — Retrieved clauses are accumulated in a per-session cache across conversation turns, with exact-match deduplication. This means the model retains access to relevant context from earlier in the conversation, not just the current turn's retrieval results. The cache is capped at 30 clauses to protect the context window.
 
4. **Generation** — The cached clauses and conversation history are passed to the LLM, which generates an answer grounded in the retrieved contract text. The system prompt instructs the model to be honest when the answer is not available rather than hallucinating.
 
### Conversation Management
 
The system maintains per-session conversation history using a sliding window of the 10 most recent turns (20 messages). This balances context retention with the model's context window limits. The session ID is returned to the client on the first request and passed back on subsequent requests to maintain continuity.
 
## Tech Stack
 
- **LLM**: Qwen3:32b via [Ollama](https://ollama.com)
- **Vector database**: [ChromaDB](https://www.trychroma.com) (persistent local storage)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com)
- **Frontend**: Basic HTML, CSS, and JavaScript, built using Claude Code
- **Chunking**: LangChain `RecursiveCharacterTextSplitter` with tiktoken `cl100k_base` encoding (256-token chunks, 80-token overlap)
- **Dataset**: [CUAD v1](https://www.atticusprojectai.org/cuad) — 510 contracts from SEC EDGAR filings
 
## Setup
 
### Prerequisites
 
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- The Qwen3:32b model pulled locally: `ollama pull qwen3:32b`
- The CUAD dataset downloaded and extracted to `cuad/data/`
 
### Installation
 
```bash
pip install fastapi uvicorn chromadb ollama langchain-text-splitters tiktoken pydantic
```
 
### Ingest contracts
 
Run the chunking script to load contracts into ChromaDB. This only needs to be done once. The script checks whether the collection is already populated before inserting.
 
```bash
python chunker.py
```
 
### Start the server
 
```bash
uvicorn app:app --reload --port 8000
```
 
### Usage
 
Send a POST request to `/chat/` with a question:
 
```bash
# First message (no session ID)
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "What are the parties to the Azul Sa Maintenance Agreement?"}'
 
# Follow-up (include session_id from previous response)
curl -X POST "http://localhost:8000/chat/?session_id=YOUR_SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "What are the key obligations of each party?"}'
```
 
## Testing and Results

### Query Testing

The CUAD dataset contains question-and-answer pairs for each contract that can be used for testing. The query retrieval system was tested using these pairs. A random sample of up to 5 questions per contract was selected. Each question was categorized as either "factual" or "conceptual", to separately measure the accuracy of query retrieval with respect to factual questions (like the name of the agreement, the names of parties, etc.) and more conceptual questions (like explaining a contract's termination provisions, events of default, etc.). Each question was submitted to the query retrieval system, and the results were checked to determine whether they contained the reference answer from the CUAD dataset.

The query testing pipeline was run 10 times to measure how the accuracy differed when retrieving 1-10 results. The results were as follows:

Conceptual accuracy when receiving 1 results: 0.24302788844621515
Factual accuracy when receiving 1 results: 0.25872093023255816
Conceptual accuracy when receiving 2 results: 0.35358565737051795
Factual accuracy when receiving 2 results: 0.3963178294573643
Conceptual accuracy when receiving 3 results: 0.4262948207171315
Factual accuracy when receiving 3 results: 0.48546511627906974
Conceptual accuracy when receiving 4 results: 0.4721115537848606
Factual accuracy when receiving 4 results: 0.5445736434108527
Conceptual accuracy when receiving 5 results: 0.5049800796812749
Factual accuracy when receiving 5 results: 0.5930232558139535
Conceptual accuracy when receiving 6 results: 0.5398406374501992
Factual accuracy when receiving 6 results: 0.6327519379844961
Conceptual accuracy when receiving 7 results: 0.5637450199203188
Factual accuracy when receiving 7 results: 0.6618217054263565
Conceptual accuracy when receiving 8 results: 0.5856573705179283
Factual accuracy when receiving 8 results: 0.6841085271317829
Conceptual accuracy when receiving 9 results: 0.6145418326693227
Factual accuracy when receiving 9 results: 0.7093023255813954
Conceptual accuracy when receiving 10 results: 0.6324701195219123
Factual accuracy when receiving 10 results: 0.7257751937984496

Based on these findings, I decided to set the agent up to retrieve 10 results per query.

### Generation Testing

The agent's generation was also tested using these question-answer pairs. A sample of 51 contracts was selected, and one question from each contract was randomly selected. Each selected question went through the entire pipeline, from query retrieval to answer generation. Then, a separate instance of Qwen 3:32b was used to compare the RAG agent's answer to the reference answer in the CUAD dataset. This instance of Qwen rated the candidate answer on a 1-5 scale based on its correctness and completeness, and provided a one-sentence rationale for each rating. The resulting distribution of scores was as follows:

5 - 44 questions
3 - 6 questions
2 - 1 question

These results suggest that the agent is able to provide accurate answers to a broad range of questions. I will note that I think LLM's can sometimes be overly generous in this context. However, based on the grading model's rationales, it appears to have taken a pretty strict approach. For example, in a couple of instances it gave a candidate answer a score of 3 because, while the answer correctly answered the question, it included additional discussion of the contract that went beyond the scope of the question. This shows that the grading model was thinking critically about the correctness and completeness of the answers, rather than simply giving higher scores to longer answers. All in all, I was encouraged by these generation test results.

### Manual Testing

The system was manually tested across five multi-turn scenarios designed to evaluate different capabilities:
 
**Cross-contract comparison** — Asked to compare party obligations and termination rights between the Azul Sa and Bloom Energy maintenance agreements across four turns. The model correctly retrieved and contrasted clauses from both contracts, producing a detailed side-by-side analysis.
 
**Ambiguous pronoun resolution** — Asked four increasingly vague follow-up questions about the Range Resources Transportation Agreement (e.g., "Can they assign it to a third party?", "What's the governing law?", "Does it allow termination for convenience?"). The query rewriting step correctly resolved pronouns back to the original contract across all turns.
 
**Hallucination resistance** — Asked about revenue-sharing and MFN clauses in the Emmis Communications Marketing Agreement (clauses unlikely to exist in that contract type), and non-compete restrictions in the Coral Gold Consulting Agreement. The model correctly reported when clauses were absent rather than fabricating answers.
 
**Topic switching** — After two turns on the Zogenix Distributor Agreement, switched to asking about a different contract and clause type. The model pivoted to the new context without old clauses bleeding into the response.
 
**Deep follow-up chain** — This scenario exposed a retrieval limitation (see below).
 
## Known Limitations and Future Work
 
### Retrieval misses on entity matching
 
The most significant limitation is that pure semantic search occasionally fails to retrieve chunks from a specifically named contract. In testing, queries about the Suntron Corp Maintenance Agreement and the Know Labs IP Agreement returned chunks from other contracts with semantically similar content, but not from the requested contract itself.
 
The root cause is that embedding-based retrieval optimizes for topical similarity rather than entity matching. When you search for "termination provisions in the Suntron Corp Maintenance Agreement," the embedding captures "termination provisions" and "maintenance agreement" strongly, but treats "Suntron Corp" as less semantically meaningful. Chunks from other maintenance agreements with similar termination language can then outrank the Suntron chunks.

The good thing about this is that when it happens, the model is upfront about it. It will not present information from another contract as if it were from the contract it was asked about. Rather, it will state upfront that the retrieved clauses do not include any provisions from the contract you asked about, but that similar provisions from other contracts were included, and that it is happy to provide information about those if helpful.
 
Potential solutions, roughly in order from most to least complex:
 
- **Metadata filtering** — Extract the contract name from the user's query and use ChromaDB's `where` clause to restrict retrieval to that contract before ranking. This guarantees entity-level precision when a specific contract is named, but it could prove difficult to reliably extract the contract name.
- **Hybrid search** — Combine vector search with keyword-based retrieval and merge results using Reciprocal Rank Fusion. Semantic search handles topical relevance, while keyword search handles exact-entity matching.
- **Cross-encoder re-ranking** — Add a second-stage re-ranker (e.g., a cross-encoder model) that jointly evaluates the query and each candidate chunk. This captures finer-grained relevance than the bi-encoder used for initial retrieval.
 
### Other areas for improvement
 
- **Clause cache eviction** — The current cache uses a simple size cap (most recent 30 clauses). A smarter strategy would re-rank cached clauses by relevance to the current query and drop the least relevant ones, rather than the oldest.
- **Persistent session storage** — Conversation history and clause caches are currently stored in memory. A production deployment would need persistent storage (e.g., Redis or a database) to survive server restarts.
- **Streaming** - Currently, when a user submits a prompt, a message appears indicating the agent is thinking. Then, once the agent has fully prepared its answer, it appears on the page all at once. UX could be improved by adding real-time token streaming, as is common in frontier LLM models.
 
## License
 
The CUAD dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying contracts are publicly available from [SEC EDGAR](https://www.sec.gov/edgar).