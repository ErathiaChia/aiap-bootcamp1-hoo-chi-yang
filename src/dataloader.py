import sqlite3 as db
import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def import_data(self):
        if os.path.isfile(self.file_path):
            print("File exists")
        else:
            print("File not found")
        
        conn = db.connect(self.file_path)
        df = pd.read_sql_query("SELECT * FROM score", conn)
        conn.close()
        print(df.head())
        return df