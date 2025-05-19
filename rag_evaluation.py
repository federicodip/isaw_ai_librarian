import os
import json
import shutil
import re
import csv
# import matplotlib.pyplot as plt
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Load ISAW papers chunks as meaningful documents with metadata
file_path = "chunks_isaw_papers_all.txt"

def extract_source(chunk):
    source_marker = "Source: "
    if source_marker in chunk:
        source_start = chunk.rfind(source_marker) + len(source_marker)
        return chunk[source_start:].strip()
    return "Unknown"

documents = []
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()
    chunks = json.loads(content)
    for chunk in chunks:
        source = extract_source(chunk)
        documents.append(Document(page_content=chunk, metadata={"source": source}))

# Initialize embeddings
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Chroma Vectorstore
persist_directory = './docs/chroma'
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=persist_directory
)

# Initialize ChatOpenAI model
llm = ChatOpenAI(model_name='gpt-4', temperature=0.3)

# Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True
)

# Load your ground truth QA pairs (your test dataset)
with open('qa_fail_rag.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# GPT-4 function to evaluate answers
def evaluate_reference_guided_grading(question, correct_answer, model_answer):
    evaluation_prompt = (
        f"Evaluate the following model answer on reference_guided_grading compared to the correct answer. "
        f"Provide a numeric reference_guided_grading score from 1 (completely inaccurate) to 10 (completely accurate). "
        f"Return only the number.\n\n"
        f"Question: {question}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Model Answer: {model_answer}\n\n"
        "reference_guided_grading Score (1-10):"
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0,
        max_tokens=10
    )
    raw_output = response.choices[0].message.content.strip()
    match = re.search(r"\b([1-9]|10)\b", raw_output)
    if match:
        return int(match.group(1))
    else:
        print(f"\u26a0\ufe0f Unexpected score format: '{raw_output}'")
        return 0

# Prepare or resume evaluation lists
results = []
tp_rag, fp_rag = 0, 0
tp_base, fp_base = 0, 0
reference_guided_grading_scores_rag = []
reference_guided_grading_scores_base = []

# Resume from existing file if available
progress_file = "partial_results_fail_rag.json"
start_index = 0
if os.path.exists(progress_file):
    with open(progress_file, 'r', encoding='utf-8') as pf:
        results = json.load(pf)
        start_index = len(results)
        print(f"\u27A1\ufe0f Resuming from index {start_index}")

# Baseline LLM without RAG
baseline_llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

for i, qa_pair in enumerate(test_data[start_index:], start=start_index):
    question = qa_pair['instruction']
    correct_answer = qa_pair['output']

    # RAG model answer
    result_rag = qa_chain({"query": question})
    model_answer_rag = result_rag['result']
    score_rag = evaluate_reference_guided_grading(question, correct_answer, model_answer_rag)
    reference_guided_grading_scores_rag.append(score_rag)
    if score_rag > 5:
        tp_rag += 1
    else:
        fp_rag += 1

    # Baseline LLM answer (no RAG)
    baseline_response = baseline_llm.predict(question)
    score_base = evaluate_reference_guided_grading(question, correct_answer, baseline_response)
    reference_guided_grading_scores_base.append(score_base)
    if score_base > 5:
        tp_base += 1
    else:
        fp_base += 1

    # Store result
    result_entry = {
        "question": question,
        "correct_answer": correct_answer,
        "rag_answer": model_answer_rag,
        "baseline_answer": baseline_response,
        "score_rag": score_rag,
        "score_baseline": score_base
    }
    results.append(result_entry)

    # Save progress every 5 items
    if i % 5 == 0:
        with open(progress_file, 'w', encoding='utf-8') as pf:
            json.dump(results, pf, ensure_ascii=False, indent=2)
        print(f"✅ Progress saved at index {i}")

# Final save
with open(progress_file, 'w', encoding='utf-8') as pf:
    json.dump(results, pf, ensure_ascii=False, indent=2)
print("🧾 Final results saved to partial_results.json")

# If no new questions were processed, exit early
if len(reference_guided_grading_scores_rag) == 0:
    print("✅ All questions already processed. No new evaluations to compute.")
    exit()


# Calculate metrics
acceptability_ratio_rag = tp_rag / (tp_rag + fp_rag) if (tp_rag + fp_rag) > 0 else 0
acceptability_ratio_base = tp_base / (tp_base + fp_base) if (tp_base + fp_base) > 0 else 0
average_reference_guided_grading_rag = sum(reference_guided_grading_scores_rag) / len(reference_guided_grading_scores_rag)
average_reference_guided_grading_base = sum(reference_guided_grading_scores_base) / len(reference_guided_grading_scores_base)

# Print results
print("--- Evaluation Summary ---")
print(f"RAG - Average reference_guided_grading: {average_reference_guided_grading_rag:.2f}, acceptability_ratio: {acceptability_ratio_rag:.2f}")
print(f"Baseline - Average reference_guided_grading: {average_reference_guided_grading_base:.2f}, acceptability_ratio: {acceptability_ratio_base:.2f}")

# Export detailed results to CSV
csv_path = "rag_vs_baseline_results_fail_rag.csv"
with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Detailed results saved to {csv_path}")
