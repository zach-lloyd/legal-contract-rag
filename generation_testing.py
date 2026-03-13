import json
import chromadb
import random
import re
import ollama
from datetime import datetime

# The number of results for the query to retrieve. Test values from 1 to 10
NUM_RESULTS = 10

# I divided the question categories into two buckets: conceptual and factual, 
# so I could measure how query performance differs between them
CONCEPTUAL_CATEGORIES = [
    'Affiliate License-Licensee', 
    'Volume Restriction',
    'Covenant Not To Sue',
    'Source Code Escrow',
    'Joint Ip Ownership',
    'Non-Transferable License',
    'Competitive Restriction Exception',
    'Non-Disparagement', 
    'Ip Ownership Assignment',
    'No-Solicit Of Employees',
    'Non-Compete', 
    'Minimum Commitment', 
    'Liquidated Damages', 
    'License Grant',
    'Cap On Liability', 
    'Change Of Control', 
    'Audit Rights', 
    'Insurance',
    'Uncapped Liability', 
    'Termination For Convenience', 
    'Post-Termination Services',
    'No-Solicit Of Customers', 
    'Most Favored Nation', 
    'Price Restrictions', 
    'Renewal Term',
    'Third Party Beneficiary', 
    'Irrevocable Or Perpetual License', 
    'Anti-Assignment', 
    'Exclusivity',
    'Unlimited/All-You-Can-Eat-License', 
    'Affiliate License-Licensor', 
    'Rofr/Rofo/Rofn', 
    'Revenue/Profit Sharing'
]

FACTUAL_CATEGORIES = [
    'Effective Date',
    'Warranty Duration',
    'Agreement Date',
    'Parties',
    'Expiration Date',
    'Document Name',
    'Notice Period To Terminate Renewal',
    'Governing Law', 
]

with open("cuad/data/train_separate_questions.json", "r") as f:
    cuad = json.load(f)

def get_qa_pairs(data):
    """
    Get question/answer pairs from each contract for use in testing.

    Parameters:
        data (dict): A JSON dictionary representing the contract dataset.

    Returns:
        dict: A dictionary of questions and answers.
    """
    qas = []

    random.seed(42)

    for i in range(0, 408, 8):
        contract = data["data"][i]
        contract_title = contract["title"]
        # Filter out any questions that are impossible to answer
        question_list = [q for q in contract["paragraphs"][0]["qas"] if not q["is_impossible"]]
        
        qa_index = random.randint(0, len(question_list) - 1)

        question = question_list[qa_index]["question"]
        answer = question_list[qa_index]["answers"][0]["text"]

        # Each question's category is surrounded by quotation marks; this code will
        # extract it
        category = re.findall(r'"([^"]*)"', question)
        category_type = ""

        if category[0] in CONCEPTUAL_CATEGORIES:
            category_type = "conceptual"
        else:
            category_type = "factual"
        
        qas.append({
            "title": contract_title, 
            "category_type": category_type,
            "question": question, 
            "answer": answer,
        })
    
    return qas

def get_candidate_answers(qas, number_of_results):
    """
    Use an LLM to generate an anwswer for each question in qas.

    Parameters:
        qas (dict): A dictionary of question/answer pairs, the title of the
                    contract they relate to, and the category type of the
                    question (factual or conceptual).
        number_of_results (int): The number of results to retrieve with each query.

    Returns:
        dict: A revised version of qas with an extra key/value pair for each question
              storing the generated answer to that question.
    """    
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_or_create_collection(name="legal_contracts")
    question_num = 1

    for qa in qas:
        print(f"Generating answer for question number {question_num}...\n")

        question = qa["question"]

        results = collection.query(
            query_texts=[question],
            n_results=number_of_results,
            # Be sure to only search the applicable contract, not the entire
            # database
            where={"contract_title": qa["title"]}
        )

        context = ""

        for i in range(len(results["documents"])):
            titles = " ".join([metadata["contract_title"] for metadata in results["metadatas"][i]])
            result = titles + " ".join(results["documents"][i])
            context += result
        
        sys_prompt = """
             You are a helpful legal assistant. Provide a concise but 
             thorough answer to the user's latest question, using only
             the existing conversation history (if any) and the relevant clauses 
             from existing legal contracts provided below. If the answer is not 
             available. You should be honest about that. Do not hallucinate a 
             false answer. Rather, you should respond 'I don't have information about that.'
             """
        
        full_sys_prompt = f"""
                  {sys_prompt} The relevant contract clauses are as follows: {context}.
                  The prior history of this conversation is below, ending with the
                  user's current question. If there is no text provided other than a
                  question from the user, then this is the first question and there 
                  is no conversation history to reference.
                  """
        
        sys_message = {"role": "system", "content": full_sys_prompt}
        q = {"role": "user", "content": f"Question: {question}"}
        
        response = ollama.chat(
            model="qwen3:32b",
            messages=[sys_message, q]
        )

        a = response["message"]["content"]
        qa["generated_answer"] = a
        question_num += 1
                    
    return qas

def score_answers(qas):
    """
    Use an LLM to compare the generated answers to the reference answers and 
    score them for correctness and completeness.

    Parameters:
        qas (dict): A dictionary of question/answer pairs, the title of the
                    contract they relate to, and the category type of the
                    question (factual or conceptual).
    
    Returns:
        dict: A JSON object containing the questions, answers, categories, a 
              score for each response, and the LLM's rationale for the score.
    """
    scores = []
    question_num = 1

    scoring_prompt = f"""
                Given the below question, reference answer, and candidate answer, 
                rate the candidate on a 1-5 scale for correctness and completeness. 
                Return only a JSON object with your scores and a one-sentence 
                rationale, using the keys "score" for your score and "rationale" 
                for your rationale.
                """

    for qa in qas:
        print(f"Scoring question number {question_num}...\n")

        question = qa["question"]
        reference_answer = qa["answer"]
        candidate_answer = qa["generated_answer"]
        category_type = qa["category_type"]

        score = ollama.chat(
            model="qwen3:32b",
            messages=[
                {"role": "system", "content": scoring_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "user", "content": f"Reference Answer: {reference_answer}"},
                {"role": "user", "content": f"Candidate Answer: {candidate_answer}"}
            ]
        )

        print(score["message"]["content"])

        scores.append({
            "question": question,
            "candidate_answer": candidate_answer,
            "reference_answer": reference_answer,
            "category_type": category_type,
            "result": score["message"]["content"]
        })

        question_num += 1
    
    # Include the timestamp in the filepath so that prior runs are not overwritten
    # and can be compared
    file_path = f"generation_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(file_path, "w") as json_file:
        json.dump(scores, json_file, indent=4)
    
    return scores

if __name__ == "__main__":
    qa_pairs = get_qa_pairs(cuad)
    candidate_answers = get_candidate_answers(qa_pairs, NUM_RESULTS)
    results = score_answers(candidate_answers)
