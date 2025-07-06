import pandas as pd
import numpy as np
import uuid
import os

files = [
#   'YG-1 JAPAN.csv',
#   'Qingdao WANJI.csv',
#   'YG-1 Technology Center.csv',
#   'Herramientas YG-1.csv',
#   'YG-1 Vietnam.csv',
#   'YG-1 Tools Asia.csv',
#   'YG-1 Eastern Europe.csv',
#   'YG-1 Poland.csv',
#   'YG-1 Russia.csv',
#   'YG-1 MALAYSIA.csv',
#   'QYG-1.csv',
#   'YG-1 Australia.csv',
#   'YG-1 Shanghai.csv',
#   'YG-1 America.csv',
#   'YG-1 Tool Solutions.csv',
#   'YG-1 Deutschland.csv',
#   'YG-1 EUROPE.csv',
#   'YG-1 Comercio.csv',
#   'YG-1 Turkey.csv',
#   'YG-1 France.csv',
#   'PT YGI TOOLS.csv',
#   'YG-1 Canada.csv',
#   'YG-1 Mexico.csv',
#   'YG INDIA.csv',
#   'YG-1 China.csv',
#   'YG-1 South Africa.csv',
#   'YG-1 THAILAND.csv',
#   'YG-1 Middle East.csv'
]

def de_identified(df):
    df_summarize = df[['Team', 'Company_code', 'Company', 'Country', 'Customer', \
                       'YYYYMMDD', 'Sales_Qty', 'Sales_Amount_USD', 'Currency','Code',\
                        'EDP_No', 'Size', 'Division', 'Product_hierarchy', 'SERIES','Description',\
                        'Product_hierarchy_Description','Created_on']]
    
    # USD로 통일
    df_summarize['Currency'] = "USD"
    df_summarize.rename(column = {"Sales_Amount_USD":"Amount"},inplace = True)
    column_list = ['Team', 'Company_code', 'Company', 'Country', 'Customer','EDP_No','Product_hierarchy', 'SERIES','Description', 'Product_hierarchy_Description']
    
    for col in column_list:
        unique_value = df_summarize[col].unique()
        uuid_map = {val: str(uuid.uuid4()) for val in unique_value}
        df_summarize[col] = df_summarize[col].map(uuid_map)



if __name__ == "__main__":
    #C:\Users\parks\Documents\cornerstp\yg1-sale-plan-backend\src\oversea
    dirs = os.listdir("./oversea")
    dirs = [ x for x in dirs if x.endswith('.csv')]
    # dirs = files

    for file in dirs:
        df = pd.read_csv(dirs)
        de_identified(df)