from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
import hashlib
import pinecone
import os
import tiktoken

tokenizer = tiktoken.get_encoding('cl100k_base')
batch_limit = 100

def tiktoken_len(text: str) -> int:
    """ Returns the number of tokens in the given text """
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_files_in_folder(folder_path: str) -> list[str]:
    """ Returns a list of filenames in the given folder """
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def pdf_to_text(pdf_path: str) -> str:
    """ Returns the text from a PDF file """
    pdf_reader = PdfReader(pdf_path)
    # read data from the file and put them into a variable called text
    text = ''
    for _, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            text += text
    return text

def hash_string(input_string: str) -> str:
    """ Returns the sha256 hash of the given string """

    # Create a new sha256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes of the input string
    sha256_hash.update(input_string.encode())

    # Get the hexadecimal representation of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest

# init embeddings model
embed = OpenAIEmbeddings(
    model='text-embedding-ada-002',
)

# init text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 32,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)


# init pinecone
PINECONE_API_KEY = os.getenv('PINECONE_RAGDEMO_API_KEY') or 'PINECONE_RAGDEMO_API_KEY'
PINECONE_ENVIRONMENT = os.getenv('PINECONE_RAGDEMO_ENV') or 'PINECONE_RAGDEMO_ENV'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

index_name = 'rag-demo'
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
index = pinecone.GRPCIndex(index_name)


# Replace '/path/to/folder' with the path to your folder
folder_path = 'documents'
files = get_files_in_folder(folder_path)

for file in files:
    text = pdf_to_text(os.path.join(folder_path, file))
    chunks = text_splitter.split_text(text)

    metadata: dict[str, str] = {
        'title': file,
        'source': file
    }

    file_id = hash_string(str(metadata))

    # create individual metadata dicts for each chunk
    metadatas: dict[str, str | int] = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(chunks)]

    # embed chunks
    embeddings = embed.embed_documents(chunks)

    ids = [f"{file_id}-{j}" for j in range(len(chunks))]
    print(list(zip(ids, embeddings, metadatas)))
    index.upsert(vectors=list(zip(ids, embeddings, metadatas)))

    print(embeddings)
    index.describe_index_stats()