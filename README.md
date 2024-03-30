# Document Question-Answering chatbot

This project is a chatbot powered by the GPT 3.5 from OpenAI. It allows users to upload PDF files and ask questions related to the content of the PDFs. The chatbot uses vector storage and retrieval techniques to search for relevant information within the uploaded documents and generate accurate responses.

## Demo

https://github.com/mosheragomaa/Document_QA_LangChain_GPT/assets/76535465/e348e906-e58d-4a7d-a490-7890ca15a3ba


## Technologies and Tools:
- **LangChain**  for text processing, document loading, and building the question-answering chain.
- **GPT 3.5** for the large language model.
- **OpenAI** Embeddings for text embeddings.
- **Chroma** as the embedding database.
- **Streamlit** for the user interface.

## Installation

- Clone the repository:

` git clone https://github.com/mosheragomaa/Document_QA_LangChain_GPT.git `

` cd Document_QA_LangChain_GPT `


- Install the required dependencies:

` pip install -r requirements.txt ` 

> [!NOTE]
> To run this project, you will need to create an OpenAI API key, and add it to the code files as follows:

1) Open  **streamlit.py:** and replace _**api_key**_ value with your API in the following code as follows:
   ``` python
   if "llm_model" not in st.session_state:
     st.session_state["llm_model"] = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key= "YOUR_API_KEY")

2) Open **helpers.py:** and replace _**api_key**_ value with your API as follows:
  

``` python
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key= "YOUR_API_KEY")

```

## Usage

1. Run the Streamlit app: ` streamlit run streamlit.py `

2. Upload one or more PDF files using the file uploader.

3. Once the PDFs are uploaded, you can start asking questions about their content in the chat interface.

The chatbot will generate responses based on the relevant information found in the uploaded PDFs.

## File Structure
- _**streamlit.py:**_ The main Streamlit app file that handles user interaction and file uploads.

- _**helpers.py:**_ Contains helper functions for loading PDFs, splitting text, creating vector stores, and building the question-answering chain.

- _**requirements.txt:**_ List of required Python packages.

## Contributions

Contributors are welcome to add: 
- Chat history feature.
- Summarization feature.
- Feature to provide document resources.
