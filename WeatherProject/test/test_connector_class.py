import unittest
import os
import sqlite3
import sys
import tempfile

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
        if os.path.exists(cls.test_db_name):
            os.remove(cls.test_db_name)
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
            train_start='2000-01-01', train_end='2007-12-01', test_start='2008-03-01', test_end='2010-03-01',
            seasonal_naive_mse=0.2, seasonal_naive_mae=0.1, sarima_mse=0.15, sarima_mae=0.08,
            sarima_order=(1, 1, 1), sarima_seasonal_order=(1, 1, 1, 12),
            model_path='models/test.pkl', code_version='code-hash', data_hash='data-hash',
            evaluation_folds=5, forecast_horizon_months=12, selected_model='sarima'
        )

        conn = self.connector.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM arima_model;")
            rows = cursor.fetchall()
            column_names = [column[0] for column in cursor.description]
        finally:
            conn.close()

        self.assertEqual(len(rows), 1)  # Ensure data was inserted
        self.assertEqual(rows[0][1], 1.0)  # adf_stat
        self.assertEqual(rows[0][2], 0.01)  # adf_pvalue
        self.assertEqual(rows[0][3], 1.0)  # adf_stat_diff
        self.assertEqual(rows[0][4], 0.01)  # adf_pvalue_diff
        self.assertEqual(rows[0][5], 5)  # order_p
        self.assertEqual(rows[0][6], 1)  # order_d
        self.assertEqual(rows[0][7], 0)  # order_q
        self.assertEqual(rows[0][8:11], (5, 1, 0))  # arima_order_p/d/q
        self.assertEqual(rows[0][11], 10)  # forecast_steps
        self.assertEqual(rows[0][12], 'Test Summary')  # model_summary
        self.assertEqual(rows[0][13], 0.1)  # mse
        self.assertEqual(rows[0][14], 0.05)  # mae
        self.assertEqual(rows[0][15], 'London')  # city
        self.assertEqual(rows[0][16], '2000-01-01')  # train_start
        self.assertEqual(rows[0][17], '2007-12-01')  # train_end
        self.assertEqual(rows[0][18], '2008-03-01')  # test_start
        self.assertEqual(rows[0][19], '2010-03-01')  # test_end
        self.assertEqual(rows[0][20], 0.2)  # seasonal_naive_mse
        self.assertEqual(rows[0][22], 0.15)  # sarima_mse
        self.assertEqual(rows[0][30], 12)  # sarima seasonal period
        self.assertEqual(rows[0][31], 'models/test.pkl')  # model_path
        self.assertEqual(rows[0][32], 'code-hash')  # code_version
        self.assertEqual(rows[0][33], 'data-hash')  # data_hash
        self.assertEqual(rows[0][34], 5)  # evaluation_folds
        self.assertEqual(rows[0][35], 12)  # forecast_horizon_months
        self.assertEqual(rows[0][36], 'sarima')  # selected_model
        self.assertIn('arima_order_p', column_names)
        self.assertIn('data_hash', column_names)

        conn.close()

    def test_create_table_migrates_existing_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "legacy.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE arima_model (id INTEGER PRIMARY KEY, "
                "order_p INTEGER, order_d INTEGER, order_q INTEGER)"
            )
            conn.execute(
                "INSERT INTO arima_model (order_p, order_d, order_q) VALUES (5, 1, 3)"
            )
            conn.commit()
            conn.close()

            SQLiteConnector(db_path).create_table()

            conn = sqlite3.connect(db_path)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(arima_model)")}
            migrated_order = conn.execute(
                "SELECT arima_order_p, arima_order_d, arima_order_q FROM arima_model"
            ).fetchone()
            conn.close()
        self.assertIn("model_path", columns)
        self.assertIn("sarima_mse", columns)
        self.assertIn("evaluation_folds", columns)
        self.assertIn("selected_model", columns)
        self.assertEqual(migrated_order, (5, 1, 3))

    def test_insert_errors_are_raised(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            connector = SQLiteConnector(os.path.join(temp_dir, "missing-table.db"))
            with self.assertRaises(sqlite3.OperationalError):
                connector.insert_model_results(
                    adf_stat=1.0, adf_pvalue=0.01, adf_stat_diff=1.0, adf_pvalue_diff=0.01,
                    order=(1, 1, 0), forecast_steps=1, model_summary='Summary', mse=0.1, mae=0.1,
                    city='London', train_start='2000-01', train_end='2000-01',
                    test_start='2000-02', test_end='2000-02'
                )


if __name__ == '__main__':
    unittest.main()
