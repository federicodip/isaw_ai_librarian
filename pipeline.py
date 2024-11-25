import os
import shutil
import json
import requests
from bs4 import BeautifulSoup
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------DOCUMENT LOADING--------------------------------#

def download_pdf_from_html(page_url, output_dir, pdf_names):
    """
    Download PDFs from the given HTML page and save to the output directory.
    """
    try:
        response = requests.get(page_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error accessing {page_url}: {e}")
        return False

    soup = BeautifulSoup(response.text, 'html.parser')
    pdf_link_tag = soup.find('a', href=lambda href: href and '.pdf' in href)
    if pdf_link_tag:
        pdf_url = pdf_link_tag['href']
        if not pdf_url.startswith("http"):
            pdf_url = requests.compat.urljoin(page_url, pdf_url)

        pdf_filename = pdf_url.split('/')[-1].split('?')[0]
        save_path = os.path.join(output_dir, pdf_filename)

        try:
            pdf_response = requests.get(pdf_url, stream=True)
            pdf_response.raise_for_status()
            with open(save_path, 'wb') as pdf_file:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)
            print(f"PDF downloaded and saved as {save_path}")
            pdf_names.append({"filename": pdf_filename, "source_url": page_url})
            return True
        except requests.RequestException as e:
            print(f"Failed to download PDF from {pdf_url}: {e}")
    else:
        print(f"No PDF link found on page {page_url}.")
    return False

def save_pdf_names_to_file(pdf_names, file_path):
    """
    Save the names and URLs of downloaded PDFs to a file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in pdf_names:
            f.write(f'{entry["filename"]},{entry["source_url"]}\n')
    print(f"PDF names and URLs have been saved to {file_path}")

def process_pdfs(base_url, output_dir, start=1800, end=1804):
    """
    Main function to download PDFs and save their names.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_names = []

    for i in range(start, end + 1):
        page_url = f"{base_url}{i}"
        print(f"Checking page: {page_url}")
        download_pdf_from_html(page_url, output_dir, pdf_names)

    save_pdf_names_to_file(pdf_names, os.path.join(output_dir, "pdf_names.txt"))

# ------------------------------TEXT DATA PREPROCESSING--------------------------------#

def split_chunk_into_halves(chunk):
    """
    Splits a chunk into two halves using punctuation as a delimiter.
    Maintains metadata for both resulting chunks.
    """
    content = chunk["content"]
    metadata = chunk["metadata"]

    mid = len(content) // 2
    punctuation = ['.', '!', '?']
    split_index = mid
    for i in range(mid, len(content)):
        if content[i] in punctuation:
            split_index = i + 1
            break

    if split_index == mid:
        for i in range(mid, 0, -1):
            if content[i] in punctuation:
                split_index = i + 1
                break

    first_half = content[:split_index].strip()
    second_half = content[split_index:].strip()

    return [
        {"content": first_half, "metadata": metadata},
        {"content": second_half, "metadata": metadata},
    ]

def save_pdf_chunks_as_strings(pdf_path, output_json_path, source_url):
    """
    Process a PDF into text chunks and save them as a JSON file.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from {pdf_path}")

        if len(pages) > 13:
            pages = pages[3:-10]
        else:
            pages = pages[3:]

        print(f"Remaining pages after slicing: {len(pages)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            separators=[".\n\n", ".\n"]
        )
        chunks = splitter.split_documents(pages)

        print(f"Generated {len(chunks)} chunks from {pdf_path}")

        chunk_data = []

        for chunk in chunks:
            cleaned_content = chunk.page_content.replace('\n', ' ')
            metadata = {
                "source": source_url,
                "page": chunk.metadata.get('page', 'Unknown')
            }
            chunk_data.append({
                "content": cleaned_content,
                "metadata": metadata
            })

        split_chunks = []
        for chunk in chunk_data:
            if len(chunk["content"]) > 700:
                split_chunks.extend(split_chunk_into_halves(chunk))
            else:
                split_chunks.append(chunk)

        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(split_chunks, file, ensure_ascii=False, indent=4)

        print(f"Chunks have been saved to {output_json_path} as a JSON array.")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def txt_to_list_of_strings(input_txt_path):
    """
    Convert a text file to a list of strings.
    """
    try:
        with open(input_txt_path, "r", encoding="utf-8") as file:
            file_list = [line.strip().split(",") for line in file.readlines()]
        print(f"Loaded file list: {file_list}")
        return file_list
    except FileNotFoundError:
        print(f"File {input_txt_path} not found.")
        return []

def load_chunks_from_json(file_path):
    """
    Load chunks from a JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)
        print(f"Successfully loaded {len(chunks)} chunks from {file_path}")
        return chunks
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []

def convert_json_to_txt_clean(json_file, output_txt_file):
    """
    Converts a JSON file containing 'content' and 'metadata' into a text file
    with each entry formatted as a string that includes both content and source,
    ensuring metadata sections are cleaned of quotes.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        formatted_texts = []

        for entry in data:
            content = entry.get("content", "").strip()
            source = entry.get("metadata", {}).get("source", "Unknown").replace('"', '')
            page = str(entry.get("metadata", {}).get("page", "Unknown")).replace('"', '')
            combined_text = f"{content} (Source: {source}, Page: {page})"
            formatted_texts.append(combined_text)
        
        with open(output_txt_file, 'w', encoding='utf-8') as file:
            json.dump(formatted_texts, file, ensure_ascii=False, indent=4)
        
        print(f"Successfully converted JSON to text file: {output_txt_file}")
    except Exception as e:
        print(f"Error occurred: {e}")

def extract_source(chunk):
    """
    Extract the source from a chunk of text.
    """
    source_marker = "Source: "
    if source_marker in chunk:
        source_start = chunk.rfind(source_marker) + len(source_marker)
        return chunk[source_start:].strip()
    return "Unknown"

# ----------------------------------TEXT EMBEDDING------------------------------#

def initialize_embeddings_and_vectorstore(documents, persist_directory):
    """
    Initialize embeddings and vectorstore.
    """
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory
    )

    return vectordb

def initialize_qa_chain(vectordb):
    """
    Initialize the QA chain.
    """
    llm = ChatOpenAI(model_name='gpt-4', temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    return qa_chain

# ----------------------------------USER INTERFACE------------------------------#

def refine_query(question, preferences, language):
    """
    Refine the query based on user preferences and language.
    """
    question = (
        "You are a helpful AI librarian. Answer the question providing relevant information. "
        "If possible, add the source and page for your answers. " + question
    )
    if preferences:
        if "Art History" in preferences:
            question += " Answer the question with relevant information from the perspective of art history."
        if "Archaeology" in preferences:
            question += " Answer the question with relevant information from the perspective of archaeology."
        if "Ancient Literature" in preferences:
            question += " Answer the question with relevant information from the perspective of ancient literature."
        if "Epigraphy" in preferences:
            question += " Answer the question with relevant information from the perspective of an epigrapher and of epigraphy."
        if "Numismatics" in preferences:
            question += " Answer the question with relevant information from the perspective of an expert in numismatics and coinage."
        if "Papyrology" in preferences:
            question += " Answer the question with relevant information from the perspective of an expert in papyrology."
        if "Ancient Medicine" in preferences:
            question += " Answer the question with relevant information from the perspective of an expert in Ancient Medicine."

    if language == "Mandarin":
        question += " Reply in Mandarin."
    elif language == "Arabic":
        question += " Reply in Arabic."
    elif language == "Spanish":
        question += " Reply in Spanish."
    elif language == "Russian":
        question += " Reply in Russian."
    else:
        question += " Reply in English."

    return question

def chatbot_response(question, preferences, language, qa_chain):
    """
    Generate a response from the QA model.
    """
    refined_question = refine_query(question, preferences, language)
    result = qa_chain({"query": refined_question})
    return result.get("result", "No response available.")

def research_summary_response(summary, vectordb):
    """
    Retrieve relevant information based on a research summary.
    """
    k = 5
    docs = vectordb.similarity_search(summary, k=k)
    response = "\n\n".join([doc.page_content for doc in docs])
    return response if response else "No relevant information found."

def handle_inputs(question, summary, preferences, language, qa_chain, vectordb):
    """
    Handle both question answering and research summaries.
    """
    if question:
        return chatbot_response(question, preferences, language, qa_chain), None
    elif summary:
        return None, research_summary_response(summary, vectordb)
    else:
        return "No input provided.", "No input provided."

def main():
    base_url = "https://dcaa.hosting.nyu.edu/items/show/"
    output_dir = "saved_pdfs"
    output_json_path = "chunks_output.json"

    # Uncomment the lines below to run the workflow
    '''
    # Step 1: Download PDFs
    process_pdfs(base_url, output_dir)

    # Step 2: Process PDFs into chunks
    input_txt_path = os.path.join(output_dir, "pdf_names.txt")
    files = txt_to_list_of_strings(input_txt_path)

    for file_entry in files:
        file_name, source_url = file_entry
        full_path = os.path.join(output_dir, file_name)
        save_pdf_chunks_as_strings(full_path, output_json_path, source_url)

    # Step 3: Load chunks and perform further processing
    chunks = load_chunks_from_json(output_json_path)
    print(chunks)  # Display the chunks or pass them to another function
    '''

    input_json = "chunks_output.json"
    output_txt = "chunks_output.txt"
    convert_json_to_txt_clean(input_json, output_txt)

    file_path = "chunks_output.txt"
    documents = []
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        chunks = json.loads(content)

        for chunk in chunks:
            source = extract_source(chunk)
            documents.append(Document(page_content=chunk, metadata={"source": source}))

    persist_directory = './docs/chroma'
    vectordb = initialize_embeddings_and_vectorstore(documents, persist_directory)
    qa_chain = initialize_qa_chain(vectordb)

    iface = gr.Interface(
        fn=lambda question, summary, preferences, language: handle_inputs(question, summary, preferences, language, qa_chain, vectordb),
        inputs=[
            gr.Textbox(lines=2, placeholder="Ask any questions about the ISAW collections"),
            gr.Textbox(lines=5, placeholder="Paste your research summary or article abstract here"),
            gr.CheckboxGroup(
                choices=["Art History", "Archaeology", "Ancient Literature", "Epigraphy", "Numismatics", "Papyrology", "Ancient Medicine"],
                label="Refine your query (optional)",
            ),
            gr.Radio(
                choices=["English", "Mandarin", "Arabic", "Spanish", "Russian"],
                label="Select Response Language",
                value="English",
            ),
        ],
        outputs=[
            gr.Textbox(label="Answer to your question"),
            gr.Textbox(label="Relevant Information from Text Chunks"),
        ],
        title="ISAW AI Librarian",
        description=(
            "Ask our Librarian a question about the ISAW collections or paste your research summary below to get "
            "relevant information. Optionally, refine your query by selecting a perspective and choose a response language."
        ),
    )

    iface.launch(share=True)

if __name__ == "__main__":
    main()
