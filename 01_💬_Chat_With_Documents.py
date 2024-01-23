from init import init
from sidebar import sidebar
import streamlit as st
from langchain.agents import initialize_agent


def chat_with_document_page():

    init()
    sidebar()

    st.title("Chat with a document")
    st.markdown('#')

    tools = [st.session_state.retriever_tool]

    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=st.session_state.llm_model,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=st.session_state.conversational_memory,
            handle_parsing_errors=True
        )

    # Initialize chat history   
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = st.session_state.rag_agent(prompt)['output']

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    chat_with_document_page()