import pandas as pd
from sqlalchemy import create_engine
import os

def upload_data():
    os.makedirs('../database', exist_ok=True)
    db_path = os.path.abspath('../database/lending_club.db')
    engine = create_engine(f"sqlite:///{db_path}")
    file_path = '../data/raw/loan_data_2007_2014.csv' 
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл не найден")
        return
    chunk_size = 50000
    first_chunk = True
    for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunk_size, index_col=0):
        mode = 'replace' if first_chunk else 'append'
        chunk.to_sql('raw_loans', engine, if_exists=mode, index=False)
        first_chunk = False

    print(f"Готово")

if __name__ == "__main__":
    upload_data()