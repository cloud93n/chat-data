import os
import glob
from typing import List
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS


load_dotenv()


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    txt_files = glob.glob(os.path.join(source_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    csv_files = glob.glob(os.path.join(source_dir, "**/*.csv"), recursive=True)
    all_files = txt_files + pdf_files + csv_files
    return [load_single_document(file_path) for file_path in all_files]

def longest_paragraph_length(text):
    # Split the text into paragraphs
    paragraphs = text.split('\n')

    # Initialize the max_length as 0
    max_length = 0

    # Iterate over all paragraphs
    for paragraph in paragraphs:
        # If the current paragraph's length is greater than max_length
        if len(paragraph) > max_length:
            # Update max_length
            max_length = len(paragraph)

    # Return the max_length
    return max_length


def main():
    # Load environment variables
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)

    for doc in documents:
        print(longest_paragraph_length(doc.page_content))

if __name__ == "__main__":
    main()
