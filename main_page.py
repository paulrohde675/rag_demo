import streamlit as st
from init import init
from sidebar import sidebar

@st.cache
def querey_qa_chain(query):
    return st.session_state.qa_chain(query)

def main_page():

    init()
    sidebar()

    st.title("Welcome to the AI Search Engine")

    st.text_input("Search query:", value="", help="e.g. 'Explain starformation in five sentences'", key="search_query")

    if st.session_state.search_query != "":
        st.session_state.qa_answer = st.session_state.qa_chain(st.session_state.search_query)

    if st.session_state.qa_answer != "":
        st.write("Answer:")
        st.write(st.session_state.qa_answer['answer'])
        
        st.write("Sources:")
        st.write(st.session_state.qa_answer['sources'])

if __name__ == "__main__":
    main_page()