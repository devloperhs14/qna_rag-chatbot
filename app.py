import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

# load environment var
load_dotenv()

# load groq and google_api_key
groq_api_key = os.getenv("GROQ_API_KEY") 
google_api_key = os.getenv("GOOGLE_API_KEY") 


# create the app

#set title
st.title("PDF Q/A Chatbot")

# setup llm - groq and gemma
llm = ChatGroq(groq_api_key=groq_api_key, model_name = "Gemma-7b-it")

# create a prompt template
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

""" )

# create vector embedding
def vector_embedding():
    # read the data, load it into chunks, create embedding and save

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdfs") # data ingestion
        st.session_state.docs = st.session_state.loader.load() # docs / pdfs loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 200) # split the text recursively and chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #create final splits
        
        #create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # create vector embeddings
        

if st.button("Create Document Embedding"):
    vector_embedding()
    
    #inform user
    st.write("Vector Store DB is Ready")

# inputs
prompt1 = st.text_input("Ask Your PDF Related Queries:")
    
    
# create a document chain
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriver = st.session_state.vectors.as_retriever() # takes the output from session state vectors ((interface)) and display output/ give to end user
    retrieval_chain = create_retrieval_chain(retriver, document_chain) # create a new retrival chain, run in form of chain
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt1})
    
    #display response
    st.write(response['answer'])
    
    #display context using expander
    with st.expander("Response Context From Document"):
        # find relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-"*8)
            
        # is is iterator and doc will have page content with context based on similarity search
    

