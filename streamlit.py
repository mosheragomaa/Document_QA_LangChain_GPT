import streamlit as st
from config import OPENAI_API_KEY, google_api
from helpers import *

st.set_page_config(page_icon = "🤖")
st.header("✨ Gemini Powered Chatbot")

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key = google_api)

pdf = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

if pdf:
    pdf = load_pdf(pdf)
    splitted_text = text_splitter(pdf)
    doc_vector_store = vector_store(splitted_text)
    rag_question_chain = question_chain(doc_vector_store)
    chain = rag_question_chain
    list_ = [splitted_text, doc_vector_store, rag_question_chain, chain, pdf]
    for var in list_:
        if var not in st.session_state:
            st.session_state.key = var

if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = GoogleGenerativeAI(model="gemini-pro", google_api_key = google_api)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask a question!")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    stream = [chunk for chunk in chain.stream(prompt)]
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})


    


