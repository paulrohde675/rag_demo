from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from init import init
from sidebar import sidebar
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

    st.text_input("Title:", value="", key="new_file_title")
    st.text_input("Author:", value="", key="new_file_author")
    st.text_input("Year:", value="", key="new_file_year")
    st.text_input("Path:", value="", key="new_file_path")
    st.markdown('#')

    if(st.button("Add file to vector store", key="add_file_to_vector_store")):
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            text = pdf_to_text(bytes_data)

            if text:
                succes = text_to_pinecone(text, uploaded_file.name)

def pdf_to_text(bytes_data: bytes) -> str | None:
    """ Returns the text from a PDF file """

    try:
        pdf_reader = PdfReader(io.BytesIO(bytes_data))
        # read data from the file and put them into a variable called text
        text = ''
        for _, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                text += text
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

    # init text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 32,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    metadata: dict[str, str] = {
        'file_name': file_name,
        'title': st.session_state.new_file_title,
        'source': st.session_state.new_file_title,
        'author': st.session_state.new_file_author,
        'year': st.session_state.new_file_year,
        'path': st.session_state.new_file_path
    }

    file_id = hash_string(str(metadata))

    # create individual metadata dicts for each chunk
    metadatas: dict[str, str | int] = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(chunks)]

    # embed chunks
    embeddings = st.session_state.embed.embed_documents(chunks)

    ids = [f"{file_id}-{j}" for j in range(len(chunks))]
    try:
        st.session_state.vector_db_index.upsert(vectors=list(zip(ids, embeddings, metadatas)))
        st.success(f"Successfully uploaded file to vector store")
        return True
    except Exception as e:
        st.error(f"Error uploading to vector store: {e}")
        return False

if __name__ == "__main__":
    upload_files()