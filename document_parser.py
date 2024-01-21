from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
import streamlit as st

def parse_document(text: str):
    """Parse a document and extract metadata. """

    json_schema = {
        "title": "Document Metadata",
        "description": "Identifying information from a document",
        "type": "object",
        "properties": {
            "title": {"title": "title", "description": "The title of the document", "type": "string"},
            "author": {"title": "author", "description": "The last name of the main / first author of the document", "type": "string"},
            "source": {"title": "source", "description": "The source of the document, e.g. an html link", "type": "string"},
            "year": {"title": "year", "description": "The year the document was published", "type": "integer"},
        },
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting information from a document in structured formats."),
            ("human", "Use the given format to extract information from the following input: {input}")
        ]
    )

    chain = create_structured_output_chain(json_schema, st.session_state.llm_model, prompt, verbose=True)
    output = chain.run(text)
    return output
