import unittest
import os
import sqlite3

def __init__(self, db_name='data/weatherDatabase.db'):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.db_name = os.path.join(BASE_DIR, db_name)

# Now you can import the SQLiteConnector class
from src.connector_class import SQLiteConnector

class TestSQLiteConnector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up a test database
        cls.test_db_name = 'test_weather_forecast.db'
        cls.connector = SQLiteConnector(cls.test_db_name)

    @classmethod
    def tearDownClass(cls):
        # Remove the test database after tests
        try:
            os.remove(cls.test_db_name)
        except OSError:
            pass

    def test_connect(self):
        # Test connection to the database
        conn = self.connector.connect()
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_create_table(self):
        # Test table creation
        self.connector.create_table()
        conn = self.connector.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='arima_model';")
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists)
        conn.close()

    def test_insert_model_results(self):
        # Test inserting model results
        self.connector.create_table()  # Ensure table is created
        self.connector.insert_model_results(
            adf_stat=1.0, adf_pvalue=0.01, adf_stat_diff=1.0, adf_pvalue_diff=0.01,
            order=(5, 1, 0), forecast_steps=10, model_summary='Test Summary', mse=0.1, mae=0.05, city = 'London',
            train_start='2000-01-01', train_end='2007-12-01', test_start='2008-03-01', test_end='2010-03-01'
        )

        conn = self.connector.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM arima_model;")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 1.0)  # adf_stat
        self.assertEqual(rows[0][2], 0.01)  # adf_pvalue
        self.assertEqual(rows[0][3], 1.0)  # adf_stat_diff
        self.assertEqual(rows[0][4], 0.01)  # adf_pvalue_diff
        self.assertEqual(rows[0][5], 5)  # order_p
        self.assertEqual(rows[0][6], 1)  # order_d
        self.assertEqual(rows[0][7], 0)  # order_q
        self.assertEqual(rows[0][8], 10)  # forecast_steps
        self.assertEqual(rows[0][9], 'Test Summary')  # model_summary
        self.assertEqual(rows[0][10], 0.1)  # mse
        self.assertEqual(rows[0][11], 0.05)  # mae
        self.assertEqual(rows[0][12], 'London') #city
        self.assertEqual(rows[0][13], '2000-01-01')  # train_start
        self.assertEqual(rows[0][14], '2007-12-01')  # train_end
        self.assertEqual(rows[0][15], '2008-03-01')  # test_start
        self.assertEqual(rows[0][16], '2010-03-01')  # test_end
        conn.close()

if __name__ == '__main__':
    unittest.main()
