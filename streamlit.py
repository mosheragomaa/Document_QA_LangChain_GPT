import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import helpers

st.set_page_config(page_icon = "🤖")
st.header("🤖 GPT3.5 Powered Chatbot")

pdf = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)


@st.cache_resource
def pdf_processing(pdf):
        pdf = helpers.load_pdf(pdf)
        splitted_text = helpers.text_splitter(pdf)
        doc_vector_store = helpers.vector_store(splitted_text)
        rag_question_chain = helpers.question_chain(doc_vector_store)
        chain = rag_question_chain
        return chain

if pdf:
     chain = pdf_processing(pdf)

if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key='YOUR_API_KEY')
    
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


    


