# ISAW AI Librarian
![image](https://github.com/user-attachments/assets/7475a9da-254b-4cdc-a654-10fc91866f02)

### Overview
The **ISAW AI Librarian** is an end-to-end pipeline tailored for humanities scholars. It enables:
1. **Data Collection**: Scraping and downloading scholarly articles from online sources.
2. **Data Preprocessing**: Splitting text into manageable chunks with metadata.
3. **Knowledge Augmentation**: Creating an augmented retrieval system to query domain-specific knowledge.
4. **User Interface**: Deploying a user-friendly interface to retrieve and explore insights from the data.

This tool is specifically designed for researchers working in domains like history, archaeology, and related fields, facilitating easy access to their data and enhanced information retrieval.

---

### Features
- **Web Scraping**: Automatically download scholarly PDFs from web pages.
- **Data Preprocessing**: Split long texts into manageable, metadata-tagged chunks for efficient storage and querying.
- **Text Embedding and Search**: Create a searchable database using advanced vector-based embeddings.
- **QA Chat Interface**: Use an interactive Gradio interface for querying domain-specific knowledge, with options to refine queries and retrieve sources.

---

### How It Works
#### Pipeline Overview
1. **Data Collection**: Download scholarly articles from URLs.
2. **Chunking**: Split documents into manageable chunks, retaining metadata like source URLs and page numbers.
3. **Embedding**: Convert text into embeddings using `OpenAIEmbeddings` for similarity-based search.
4. **Query Interface**: Implement an AI-powered librarian using `langchain` to answer questions and retrieve information interactively.

---

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/isaw-ai-librarian.git
   cd isaw_ai_librarian
   
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Set up environment variables for OpenAI API:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key

## User Interface
The Gradio-powered interface offers:

- **Textbox**: For asking questions or inputting research summaries.
- **Refinement Options**: Use checkboxes for domain-specific perspectives like "Art History" or "Archaeology."
- **Multilingual Support**: Receive responses in English, Mandarin, Arabic, Spanish, or Russian.

---

## Example

![image](https://github.com/user-attachments/assets/ba288d6d-60c3-43a9-a425-85d6ae3d837d)

### Research Summary:
![image](https://github.com/user-attachments/assets/310273f5-9502-407d-900b-dc21c00d662b)

---

## Key Components
1. **`download_pdf_from_html()`**: Scrapes PDFs from a webpage.
2. **`split_chunk_into_halves()`**: Splits large text into smaller chunks using punctuation.
3. **`convert_json_to_txt_clean()`**: Cleans and formats JSON into readable text.
4. **`ChatOpenAI`**: GPT-4-based chat model for Q&A.
5. **Gradio Interface**: Easy-to-use UI for querying the dataset.

---

## Contributions
Contributions are welcome! Submit a pull request or open an issue to suggest improvements or report bugs.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Developed with **[LangChain](https://www.langchain.com/)** and **[Gradio](https://www.gradio.app/)**.
- My gratitude goes to Sebastian Heath and Patrick Burns of New York University, as well as Riccardo Torlone of Roma Tre University, for their guidance on this project. I also wish to acknowledge the ISAW scholars at the Institute for the Study of the Ancient World at NYU for their valuable inputs and for curating the collections.

---

Enjoy extracting knowledge from your domain-specific datasets with ease!


