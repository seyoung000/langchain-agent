from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utils import translate_text, preprocess_text
from create_database import generate_data_store
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "json")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")

PROMPT_TEMPLATE = """
You are a helpful assistant.

Use the following context if relevant.

Context:
{context}

Question:
{question}

Answer:
"""

def init_setting():
    # Prepare the DB.
    if not os.path.exists(CHROMA_PATH):
        print("🛑No vector DB found. Generating now...")
        generate_data_store()
    else:
        print("Vector DB found. Using existing index.")

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    llm = ChatOpenAI()
    
    # ConversationSummaryMemory
    memory = ConversationBufferMemory(return_messages=True, input_key="question")

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    chain = LLMChain(
        llm = llm,
        prompt = prompt,
        memory = memory
    )
    return db, chain


def multi_turn_chat():

    # Initialize the setting
    db, chain = init_setting()  

    while True:
        user_input = input("✏️  Please enter your query (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("Exiting the chat.")
            break

        # Preprocess the input text
        query_text, lang = preprocess_text(user_input)

        # Search similarity from the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if not results or results[0][1] < 0.7:
            print(f"Unable to find matching results.")
            response = chain.llm.invoke(query_text)
            sources = ["(no source: answered from model knowledge)"]

        else : 
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            response = chain.invoke({
                "context": context_text,
                "question": query_text
            })

            sources = [doc.metadata.get("url", "N/A") for doc, _score in results]

        if hasattr(response, "content"):
            response_text = response.content
        elif hasattr(response, "text"):
            response_text = response.text
        elif hasattr(response, "answer"):
            response_text = response.answer
        elif isinstance(response, dict):
            response_text = response.get("content") or response.get("text") or response.get("answer") or str(response)
        else:
            response_text = str(response)        
        formatted_response = f"Response: {response_text}\nSources:\n"+"\n".join(sources)

        # Translate the response if necessary
        if lang != 'en':
            formatted_response = translate_text(formatted_response, target_lang=lang)
        print("📖",formatted_response)


if __name__ == "__main__":
    multi_turn_chat()