from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

import streamlit as st
import pinecone
import os

def init():

    # init embeddings model
    if 'embed' not in st.session_state:
        st.session_state.embed = OpenAIEmbeddings(
            model='text-embedding-ada-002',
        )
    
    # init vector store
    if 'vector_store' not in st.session_state:
        PINECONE_API_KEY = os.getenv('PINECONE_RAGDEMO_API_KEY') or 'PINECONE_RAGDEMO_API_KEY'
        PINECONE_ENVIRONMENT = os.getenv('PINECONE_RAGDEMO_ENV') or 'PINECONE_RAGDEMO_ENV'

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )

        text_field = "text"
        index_name = 'rag-demo'

        # create index if it does not exist
        if index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=index_name,
                metric='cosine',
                dimension=1536  # 1536 dim of text-embedding-ada-002
            )

        # switch back to normal index for langchain
        index = pinecone.Index(index_name)

        st.session_state.vectorstore = Pinecone(
            index, st.session_state.embed, text_field
        )
        
        st.session_state.vector_db_index = pinecone.GRPCIndex(index_name)

    # completion llm
    if 'qa_answer' not in st.session_state:
        st.session_state.qa_answer = ""

    if 'qa_chain' not in st.session_state:
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            temperature=0.0
        )

        st.session_state.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever()
        )