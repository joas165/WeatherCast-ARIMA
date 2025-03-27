import sqlite3
import os

class SQLiteConnector:
    def __init__(self, db_name='data/weatherDatabase.db'):
        if not os.path.isabs(db_name):
            BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_name = os.path.join(BASE_DIR, db_name)

        # Ensure the directory exists before using the database file
        db_dir = os.path.dirname(db_name)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)  # Create directory if missing

        self.db_name = db_name
        print(f"Using database: {self.db_name}")  # Debugging

    def connect(self):
        """Establish a connection to the SQLite database."""
        return sqlite3.connect(self.db_name)

    def create_table(self):
        """Create the arima_model table if it does not exist."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS arima_model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                adf_stat REAL,
                adf_pvalue REAL,
                adf_stat_diff REAL,
                adf_pvalue_diff REAL,
                order_p INTEGER,
                order_d INTEGER,
                order_q INTEGER,
                forecast_steps INTEGER,
                model_summary TEXT,
                mse REAL,
                mae REAL,
                city TEXT,
                train_start TEXT,
                train_end TEXT,
                test_start TEXT,
                test_end TEXT
            )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while creating the table: {e}")
        finally:
            conn.close()

    def insert_model_results(self, adf_stat, adf_pvalue, adf_stat_diff, adf_pvalue_diff, order, forecast_steps, model_summary, mse, mae, city, train_start, train_end, test_start, test_end):
        """Insert model results into the arima_model table."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT INTO arima_model (
                adf_stat, adf_pvalue, adf_stat_diff, adf_pvalue_diff, 
                order_p, order_d, order_q, forecast_steps, model_summary, mse, mae, city,
                train_start, train_end, test_start, test_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                adf_stat, adf_pvalue, adf_stat_diff, adf_pvalue_diff, 
                order[0], order[1], order[2], forecast_steps, model_summary, mse, mae, city,
                train_start, train_end, test_start, test_end
            ))
            conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while inserting the data: {e}")
        finally:
            conn.close()

