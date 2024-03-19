from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai.llms import GoogleGenerativeAI
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

def load_pdf(pdf_files):
    '''
     """
    Loads the contents of one or more PDF files using PyPDFLoader.

    Args:
        pdf_files: A single PDF file object or a list of PDF file objects.

    Returns:
        list: A list of strings representing the contents of the PDF file(s).

    This function takes either a single PDF file object or a list of PDF file objects
    as input. It processes each PDF file separately using PyPDFLoader and returns the
    contents of the file(s) as a list of strings.

    If a single PDF file is provided, the function reads the file, creates a temporary
    file to store its contents, and then loads the PDF using PyPDFLoader. The loaded
    contents are returned as a list of strings.

    If a list of PDF files is provided, the function iterates over each file, reads its
    contents, creates a temporary file for each PDF, and loads the PDFs using PyPDFLoader.
    The loaded contents of all the PDFs are concatenated into a single list of strings and
    returned.

    '''
    if isinstance(pdf_files, list):
        # If a list of files is uploaded, process each file separately
        pdf_contents = []
        for pdf_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False) as fp:
                fp.write(pdf_file.read())
                fp.flush()
                loader = PyPDFLoader(fp.name)
                pdf_content = loader.load()
                pdf_contents.extend(pdf_content)
    else:
        # If a single file is uploaded, process it directly
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(pdf_files.read())
            fp.flush()
            loader = PyPDFLoader(fp.name)
            pdf_contents = loader.load()

    return pdf_contents

def text_splitter(pdf):
    '''
    This function splits the PDF text using the configured RecursiveCharacterTextSplitter and
    returns a list of Document objects. Each Document object represents a chunk of text from
    the PDF and contains the text content and metadata.

    Args:
        pdf (list): A list of strings representing the text content of a PDF.

    Returns:
        list: A list of Document objects, where each Document object represents a chunk
              of text from the PDF.
    '''
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=20,
    add_start_index=True, 
    separators=["\n\n"], 
    is_separator_regex=False)
    doc = text_splitter.split_documents(pdf)
    return doc

def vector_store(doc):
    '''
    Creates a vector store using Chroma and returns a retriever for similarity search.

    Args:
        doc (list): A list of Document objects representing the text chunks to be stored
                    in the vector store.

    Returns:
        Retriever: A retriever object that can be used to perform similarity search on the
                   vector store.

    This function takes a list of Document objects as input and creates a vector store using
    the Chroma library. The vector store is used to efficiently store and retrieve text chunks
    based on their vector representations.

    The function performs the following steps:
    1. Creates a Chroma vector store with the following configuration:
       - `collection_name`: The name of the collection in the vector store. Set to "Documents".
       - `documents`: The list of Document objects to be stored in the vector store.
       - `embedding`: The embedding function used to convert text into vectors. Set to OpenAIEmbeddings
                      with the provided OpenAI API key.
       - `ids`: A list of unique identifiers for each Document object, generated using UUID.
    2. Creates a retriever object from the vector store using the `as_retriever` method.
       The retriever is configured for similarity search with the following parameters:
       - `search_type`: The type of search to perform. Set to Cosine similarity.
       - `search_kwargs`: Additional keyword arguments for the search. Set to {"k": 6},
                          indicating that the top 6 most similar results should be returned.
    3. Returns the retriever object.
    '''
    vectorstore = Chroma.from_documents(collection_name="Documents",
                                    documents = doc,
                                    embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY),
                                    ids =[str(uuid.uuid4()) for _ in range(len(doc))])
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def format_docs(docs):
    '''
    Formats a list of Document objects into a single string.

    Args:
        docs (list): A list of Document objects to be formatted.

    Returns:
        str: A formatted string containing the concatenated page content of the Document objects,
             separated by double newline characters ("\n\n").
    '''
    return "\n\n".join(doc.page_content for doc in docs)

def question_chain(retriever):
    '''
    This function creates a question-answering chain by combining a retriever and a language model.
    The retriever is used to fetch relevant documents based on a given query, while the language
    model is used to generate answers based on the retrieved context.

    Args:
        retriever (Retriever): A retriever object that retrieves relevant documents based on a query.

    Returns:
        Chain: A question-answering chain that takes a question as input and returns an answer
               based on the context provided by the retriever.

    The resulting question-answering chain can be used to answer questions by providing the chain
    with a question as input. The chain will retrieve relevant documents using the retriever,
    format the documents into a single string, and generate an answer using the language model
    based on the retrieved context and the question.
    '''
    template = """
    Answer the question based only on the following context:{context}, 
    do not answer from outside the context,
    and provide evidence to your answer Question: {question}, 
    if you don't know say "I'm sorry, I didn't understand that. Could you try asking in a different way?"
    """
    
    prompt = PromptTemplate.from_template(template)
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key= google_api)
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
