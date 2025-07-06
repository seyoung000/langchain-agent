# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
from dotenv import load_dotenv
import os
import shutil

load_dotenv()


# def main():
#     generate_data_store()


def generate_data_store(file_path,chroma_path):
    #md
    # documents = load_documents()

    # json
    # json_docs = load_json()
    # chunks = split_text(json_docs)
    # save_to_chroma(chunks,PATH)

    #csv
    csv_summaries = generate_csv_summaries(file_path)
    save_to_chroma(csv_summaries, chroma_path)


def load_json():
    loader = DirectoryLoader(
        path = JSON_PATH,
        glob = "*.json",
        loader_cls = JSONLoader,
        loader_kwargs={
            "jq_schema": ".[]",
            "content_key": "html",
            "metadata_func": lambda obj, idx: {"url": obj["url"], "title": obj["title"]},
            "text_content": False
        }
    )
    documents = loader.load()
    return documents


def generate_csv_summaries(PATH):
    documents = []
    for filename in os.listdir(PATH):
        if filename.endswith(".csv"):
            print(filename)
            path = os.path.join(PATH, filename)
            try:
                df = pd.read_csv(path, nrows=2)  # Only read first 2 rows to keep it lightweight
                columns = df.columns.tolist()
                preview = df.head(2).to_dict(orient="records")
                summary = f"File: {filename}\nColumns: {columns}\nPreview: {preview}"
                documents.append(Document(page_content=summary, metadata={"filename": filename}))
            except Exception as e:
                print(f"🛑 Failed to read {filename}: {e}")
    print(f"Generated {len(documents)} CSV summaries.")
    print(documents[0].page_content)
    return documents


def load_documents():
    loader = DirectoryLoader(JSON_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document],PATH):
    # Clear out the database first.
    if os.path.exists(PATH):
        print(f"Appending to existing Chroma DB at {PATH}")
        db = Chroma(persist_directory=PATH, embedding_function=OpenAIEmbeddings())
        db.add_documents(chunks)

    else:
        # Create a new DB from the documents.
        print(f"Creating new Chroma DB at {PATH}")
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=PATH
        )

    db.persist()
    print(f"Saved {len(chunks)} chunks to {PATH}.")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    JSON_PATH = os.path.join(BASE_DIR, "data", "json")
    CSV_PATH = os.path.join(BASE_DIR, "data", "csv")
    CHROMA_PATH_JSON = os.path.join(BASE_DIR, "chroma/json")
    CHROMA_PATH_CSV = os.path.join(BASE_DIR, "chroma/csv")

    generate_data_store(CSV_PATH,CHROMA_PATH_CSV)
