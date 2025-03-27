import unittest
import os
import sqlite3
import sys

# Ensure we can import from src
TEST_DIR = os.path.dirname(__file__)  # Get the directory of this test file
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))  # Move up to the project root
SRC_DIR = os.path.join(PROJECT_ROOT, "src")  # Path to src folder

sys.path.append(SRC_DIR)  # Add src to Python path

from connector_class import SQLiteConnector 

class TestSQLiteConnector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a test database and initialize connector"""
        cls.test_db_name = os.path.join(TEST_DIR, "test_weather_forecast.db")
        cls.connector = SQLiteConnector(db_name=cls.test_db_name)
        cls.connector.create_table()  # Ensure table exists

    @classmethod
    def tearDownClass(cls):
        """Remove the test database after tests"""
        if os.path.exists(cls.test_db_name):
            os.remove(cls.test_db_name)

    def test_connect(self):
        """Test if connection to the test database works"""
        conn = self.connector.connect()
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()

    def test_create_table(self):
        """Test if the arima_model table is created properly"""
        conn = self.connector.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='arima_model';")
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists)
        conn.close()

    def test_insert_model_results(self):
        """Test inserting model results into the table"""
        self.connector.insert_model_results(
            adf_stat=1.0, adf_pvalue=0.01, adf_stat_diff=1.0, adf_pvalue_diff=0.01,
            order=(5, 1, 0), forecast_steps=10, model_summary='Test Summary', mse=0.1, mae=0.05, city='London',
            train_start='2000-01-01', train_end='2007-12-01', test_start='2008-03-01', test_end='2010-03-01'
        )

        conn = self.connector.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM arima_model;")
        rows = cursor.fetchall()

        self.assertEqual(len(rows), 1)  # Ensure data was inserted
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
        self.assertEqual(rows[0][12], 'London')  # city
        self.assertEqual(rows[0][13], '2000-01-01')  # train_start
        self.assertEqual(rows[0][14], '2007-12-01')  # train_end
        self.assertEqual(rows[0][15], '2008-03-01')  # test_start
        self.assertEqual(rows[0][16], '2010-03-01')  # test_end

        conn.close()

if __name__ == '__main__':
    unittest.main()
