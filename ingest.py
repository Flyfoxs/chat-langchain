"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader, TextLoader, UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""
    loader = TextLoader("./docs/content.txt")
    raw_documents = loader.load()
    print(f'raw_documents:{len(raw_documents)}')
    print(f'raw_documents:{raw_documents[0]}')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
