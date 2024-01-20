from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from init import init
from sidebar import sidebar
from document_parser import parse_document
import tiktoken
import hashlib
import io

def upload_files():
    """Upload files to pineconde vector base. """

    init()
    sidebar()

    st.title("Upload files to the AI Search Engine")
    st.markdown('#')

    uploaded_file = st.file_uploader("Upload files here:", type=["pdf"], key="file_uploader")
    st.markdown('#')

    st.session_state.new_file_title = ""
    st.session_state.new_file_author = ""
    st.session_state.new_file_year = ""
    st.session_state.new_file_path = ""

    input_container = st.container()

    col_left, col_right = st.columns([0.6, 0.4])
    if(col_left.button("Add file to vector store", key="add_file_to_vector_store")):
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            text = pdf_to_text(bytes_data)

            if text:
                text_to_pinecone(text, uploaded_file.name)

    if(col_right.button("Auto suggest Metadata", key="auto_suggest_metadata")):
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            text = pdf_to_text(bytes_data)

            if text:
                print(len(text))
                metadata = parse_document(text[:2500])
                if metadata['title']:
                    st.session_state.new_file_title = metadata['title']
                if metadata['author']:
                    st.session_state.new_file_author = metadata['author']
                if metadata['year']:
                    st.session_state.new_file_year = metadata['year']
                if metadata['source']:
                    st.session_state.new_file_path = metadata['source']

    with input_container:
        st.session_state.new_file_title = st.text_input("Title:", value=st.session_state.new_file_title)
        st.session_state.new_file_author = st.text_input("Author:", value=st.session_state.new_file_author)
        st.session_state.new_file_year = st.text_input("Year:", value=st.session_state.new_file_year)
        st.session_state.new_file_path = st.text_input("Path:", value=st.session_state.new_file_path)
        st.markdown('#')


def pdf_to_text(bytes_data: bytes) -> str | None:
    """ Returns the text from a PDF file """

    try:
        pdf_reader = PdfReader(io.BytesIO(bytes_data))
        # read data from the file and put them into a variable called text
        text = ''
        for _, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

def hash_string(input_string: str) -> str:
    """ Returns the sha256 hash of the given string """

    # Create a new sha256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes of the input string
    sha256_hash.update(input_string.encode())

    # Get the hexadecimal representation of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest

def tiktoken_len(text: str) -> int:
    """ Returns the number of tokens in the given text """
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def text_to_pinecone(text: str, file_name: str) -> bool:
    """ Uploads a text to the pinecone vector store. """

    print(f"Uploading file {file_name} to vector store")

    # init text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 32,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    n_chunks = len(chunks)

    # Create a progress bar
    progress_bar = st.progress(0)    
    message_placeholder = st.empty()

    for j, chunk in enumerate(chunks):

        metadata: dict[str, str | int] = {
            'file_name': file_name,
            'chunk': j,
            'title': st.session_state.new_file_title,
            'source': st.session_state.new_file_title,
            'author': st.session_state.new_file_author,
            'year': st.session_state.new_file_year,
            'path': st.session_state.new_file_path,
            'text': chunk,
        }

        file_id = hash_string(str(metadata))
        chunk_id = f"{file_id}-{j}"

        # embed chunks
        embedding = st.session_state.embed.embed_documents([chunk])[0]
        # print([embedding, chunk_id, metadata])
        print(j)

        try:
            st.session_state.vector_db_index.upsert(vectors=[(chunk_id, embedding, metadata)])
            message_placeholder.success(f"Successfully uploaded chunk {j} to vector store")
            print(f"Successfully uploaded chunk {j} to vector store")
        except Exception as e:
            message_placeholder.error(f"Error uploading to vector store: {e}")
            return False
        
        # Update the progress bar with each iteration.
        progress_bar.progress((j + 1) / n_chunks)

if __name__ == "__main__":
    upload_files()