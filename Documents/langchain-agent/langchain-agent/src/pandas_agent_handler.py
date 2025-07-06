from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import os

DATA_PATH = "data/csv"

def load_csv_file(file_path,matched_files):

    dfs = {}
    for filename in matched_files:
        full_path = os.path.join(file_path, filename)
        if os.path.exists(full_path) and filename.endswith('.csv'):
            try:
                df = pd.read_csv(full_path)
                dfs[filename] = df
            except Exception as e:
                print(f"🛑 Failed to load {full_path}: {e}")
        else:
            print(f"🛑 File {full_path} does not exist or is not a CSV file.")
    return dfs


def run_pandas_agent(llm, dfs: dict[str, pd.DataFrame], query: str):
    if len(dfs) == 1:
        # 단일 파일 분석
        fname, selected_df = list(dfs.items())[0]
        # agent = create_pandas_dataframe_agent(llm, df, verbose=True)
        # return agent.run(query)

    else:
        # 다중 파일 분석 - 파일명을 추가한 열로 병합
        selected_df = pd.concat(
            [df.assign(__source__=fname) for fname, df in dfs.items()],
            ignore_index=True
        )
    
    agent = create_pandas_dataframe_agent(llm, selected_df, verbose=True,allow_dangerous_code=True)
    response = agent.invoke({"input" : query})
    return response.get('output')


# agent = create_pandas_dataframe_agent(
#     ChatOpenAI(temperature=0, model = "gpt-3.5-turbo"),

# )

if __name__ == "__main__":
    load_csv_file()