from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from utils import translate_text, preprocess_text
from create_database import generate_data_store
from pandas_agent_handler import run_pandas_agent, load_csv_file
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "data", "json")
CSV_PATH = os.path.join(BASE_DIR, "data", "csv")
CHROMA_PATH_JSON = os.path.join(BASE_DIR, "chroma/json")
CHROMA_PATH_CSV = os.path.join(BASE_DIR, "chroma/csv")

FILE_HINT_PROMPT = """
You are a helpful assistant.
Extract country, region, or continent-related words from the given text.

Text:
{text}

Example output: ["Europe", "Asia", "France", "China"]
Output must be only a Python list.
"""

AGENT_PROMPT = """
You are a helpful assistant.

Use the following context if relevant.

Previous conversation:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

def init_setting(file_path, chroma_path):
    # Prepare the DB.
    if not os.path.exists(chroma_path):
        print("🛑No vector DB found. Generating now...")
        generate_data_store(file_path, chroma_path)
    else:
        print("Vector DB found. Using existing index.")

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    llm = ChatOpenAI()
    
    # ConversationSummaryMemory
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True, 
        input_key="question"
    )

    prompt = PromptTemplate(
        input_variables=["history","context", "question"],
        template=AGENT_PROMPT
    )

    chat_chain = LLMChain(
        llm = llm,
        prompt = prompt,
        memory = memory
    )
    return db, chat_chain, llm


def extract_file_hints(query_text,llm):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=FILE_HINT_PROMPT
    )

    hint_chain = LLMChain(
        llm = llm,
        prompt = prompt
    )

    response = hint_chain.invoke({"text": query_text})
    output = response["text"] if isinstance(response, dict) else str(response)

    try:
        hints = eval(output) if output.startswith("[") else []
        if isinstance(hints, list):
            return [str(h).lower().strip() for h in hints]
        return []
    except Exception:
        return []


def multi_turn_chat(file_path, chroma_path):

    # Initialize the setting
    db, chat_chain, llm = init_setting(file_path, chroma_path)  

    while True:
        user_input = input("✏️  Please enter your query (or type 'exit' to quit): ").strip()

        if user_input.lower() == "exit":
            print("Exiting the chat.")
            break

        # Preprocess the input text
        query_text, lang = preprocess_text(user_input)

        # Search similarity from the DB.
        keyword_list = extract_file_hints(query_text,llm)

        if keyword_list:
            print("🔍 Extracted keywords:", keyword_list)
            keywords_str = " ".join(keyword_list)
            enriched_query = f"{query_text} 관련 키워드: {keywords_str}"
        else:
            enriched_query = query_text
        results = db.similarity_search_with_relevance_scores(enriched_query, k=5)

        if not results or all(score < 0.6 for _, score in results):
            # print(f"Unable to find matching results.")
            response = chat_chain.invoke({"question": query_text, "context": ""})
            sources = ["(no source: answered from model knowledge)"]

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

        else : 
            matched_files = [doc.metadata.get("filename", "N/A") for doc, _score in results]
            print(f"🔍,{matched_files}")

            dfs = load_csv_file(file_path,matched_files)

            response_text = run_pandas_agent(llm,dfs,enriched_query)

            chat_chain.memory.chat_memory.add_user_message(query_text)
            chat_chain.memory.chat_memory.add_ai_message(response_text)
            

            # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            # response = chat_chain.invoke({
            #     "context": context_text,
            #     "question": query_text
            # })

            sources = matched_files

                
        # sources = [doc.metadata.get("url", "N/A") for doc, _score in results]            
        formatted_response = f"Response: {response_text}\nSources:\n"+"\n".join(sources)

        # Translate the response if necessary
        if lang != 'en':
            formatted_response = translate_text(formatted_response, target_lang=lang)
        print("📖",formatted_response)


if __name__ == "__main__":
    multi_turn_chat(CSV_PATH, CHROMA_PATH_CSV)
