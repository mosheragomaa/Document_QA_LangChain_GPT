# Document Question-Answering chatbot

This project is a chatbot powered by the Gemini language model from Google's Generative AI. It allows users to upload PDF files and ask questions related to the content of the PDFs. The chatbot uses vector storage and retrieval techniques to search for relevant information within the uploaded documents and generate accurate responses.

## Demo




## Technologies and Tools:
- **LangChain**  for text processing, document loading, and building the question-answering chain.
- **Gemini** for the large language model.
- **OpenAI** Embeddings for text embeddings.
- **Chroma** as the embedding database.
- **Streamlit** for the user interface.

## Installation

- Clone the repository:

` git clone https://github.com/mosheragomaa/Doc-QA-Chatbot-using-LangChain-Gemini.git `

` cd Doc-QA-Chatbot-using-LangChain-Gemini `


- Install the required dependencies:

` pip install -r requirements.txt ` 

> [!NOTE]
> To run this project, you will need to create a config.py file in the project directory and add your OpenAI API key and Google Generative AI API key to it as follows:

```python
OPENAI_API_KEY = "your-openai-api-key" 

google_api = "your-google-api-key"
```


## Usage

1. Run the Streamlit app: ` streamlit run streamlit.py `

2. Upload one or more PDF files using the file uploader.

3. Once the PDFs are uploaded, you can start asking questions about their content in the chat interface.

The chatbot will generate responses based on the relevant information found in the uploaded PDFs.

## File Structure
- _**streamlit.py:**_ The main Streamlit app file that handles user interaction and file uploads.

- _**helpers.py:**_ Contains helper functions for loading PDFs, splitting text, creating vector stores, and building the question-answering chain.

- _**config.py:**_ Configuration file for storing API keys.

- _**requirements.txt:**_ List of required Python packages.
