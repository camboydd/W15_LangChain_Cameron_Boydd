import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import sqlite3
import pysqlite3  # Import pysqlite3 to override the default sqlite3

sqlite3 = pysqlite3.dbapi2  # Override the sqlite3 module

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load and read uploaded document
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)

        # Embed the text chunks
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create a vector store (ChromaDB)
        db = Chroma.from_documents(texts, embeddings)

        # Create a retriever interface
        retriever = db.as_retriever()

        # Set up the question-answering chain
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key),
            chain_type='stuff',
            retriever=retriever
        )

        # Return the response to the query
        return qa.run(query_text)
# Set page title and header
st.set_page_config(page_title='🦜🦜 Ask the Doc App')
st.title('🦜🦜 Ask the Doc App')

# File upload widget
uploaded_file = st.file_uploader('Upload an article', type='txt')

# User query input
query_text = st.text_input(
    'Enter your question:',
    placeholder='Please provide a short summary.',
    disabled=not uploaded_file
)

# Display form for OpenAI API Key & query submission
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    submitted = st.form_submit_button(
        'Submit', 
        disabled=not (uploaded_file and query_text)
    )

    # If form submitted and API key is valid
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
        del openai_api_key

# Display the result
if len(result):
    st.info(response)
