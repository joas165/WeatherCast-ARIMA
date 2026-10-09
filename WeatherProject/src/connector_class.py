import sqlite3
import os


class SQLiteConnector:
    def __init__(self, db_name="data/weatherDatabase.db"):
        if not os.path.isabs(db_name):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            db_name = os.path.join(base_dir, db_name)

        db_dir = os.path.dirname(db_name)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.db_name = db_name
        print(f"Using database: {self.db_name}")

    def connect(self):
        """Establish a connection to the SQLite database."""
        return sqlite3.connect(self.db_name)

    def create_table(self):
        """Create or upgrade the arima_model table."""
        conn = self.connect()
        try:
            with conn:
                conn.execute("""
            CREATE TABLE IF NOT EXISTS arima_model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                adf_stat REAL,
                adf_pvalue REAL,
                adf_stat_diff REAL,
                adf_pvalue_diff REAL,
                order_p INTEGER,
                order_d INTEGER,
                order_q INTEGER,
                arima_order_p INTEGER,
                arima_order_d INTEGER,
                arima_order_q INTEGER,
                forecast_steps INTEGER,
                model_summary TEXT,
                mse REAL,
                mae REAL,
                city TEXT,
                train_start TEXT,
                train_end TEXT,
                test_start TEXT,
                test_end TEXT,
                seasonal_naive_mse REAL,
                seasonal_naive_mae REAL,
                sarima_mse REAL,
                sarima_mae REAL,
                sarima_order_p INTEGER,
                sarima_order_d INTEGER,
                sarima_order_q INTEGER,
                sarima_seasonal_p INTEGER,
                sarima_seasonal_d INTEGER,
                sarima_seasonal_q INTEGER,
                sarima_seasonal_period INTEGER,
                model_path TEXT,
                code_version TEXT,
                data_hash TEXT,
                evaluation_folds INTEGER,
                forecast_horizon_months INTEGER,
                selected_model TEXT
            )
            """)
                existing_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(arima_model)")
                }
                additions = {
                    "arima_order_p": "INTEGER",
                    "arima_order_d": "INTEGER",
                    "arima_order_q": "INTEGER",
                    "seasonal_naive_mse": "REAL",
                    "seasonal_naive_mae": "REAL",
                    "sarima_mse": "REAL",
                    "sarima_mae": "REAL",
                    "sarima_order_p": "INTEGER",
                    "sarima_order_d": "INTEGER",
                    "sarima_order_q": "INTEGER",
                    "sarima_seasonal_p": "INTEGER",
                    "sarima_seasonal_d": "INTEGER",
                    "sarima_seasonal_q": "INTEGER",
                    "sarima_seasonal_period": "INTEGER",
                    "model_path": "TEXT",
                    "code_version": "TEXT",
                    "data_hash": "TEXT",
                    "evaluation_folds": "INTEGER",
                    "forecast_horizon_months": "INTEGER",
                    "selected_model": "TEXT",
                }
                for column, column_type in additions.items():
                    if column not in existing_columns:
                        conn.execute(
                            f"ALTER TABLE arima_model ADD COLUMN {column} {column_type}"
                        )
                if {"order_p", "order_d", "order_q"}.issubset(existing_columns):
                    conn.execute("""
                        UPDATE arima_model
                        SET arima_order_p = order_p,
                            arima_order_d = order_d,
                            arima_order_q = order_q
                        WHERE arima_order_p IS NULL
                          AND arima_order_d IS NULL
                          AND arima_order_q IS NULL
                    """)
        finally:
            conn.close()

    def insert_model_results(
        self,
        adf_stat,
        adf_pvalue,
        adf_stat_diff,
        adf_pvalue_diff,
        order,
        forecast_steps,
        model_summary,
        mse,
        mae,
        city,
        train_start,
        train_end,
        test_start,
        test_end,
        seasonal_naive_mse=None,
        seasonal_naive_mae=None,
        sarima_mse=None,
        sarima_mae=None,
        sarima_order=(None, None, None),
        sarima_seasonal_order=(None, None, None, None),
        model_path=None,
        code_version=None,
        data_hash=None,
        evaluation_folds=None,
        forecast_horizon_months=None,
        selected_model=None,
    ):
        """Insert model results into the arima_model table."""
        conn = self.connect()
        try:
            with conn:
                conn.execute("""
            INSERT INTO arima_model (
                adf_stat, adf_pvalue, adf_stat_diff, adf_pvalue_diff,
                order_p, order_d, order_q, arima_order_p, arima_order_d, arima_order_q,
                forecast_steps, model_summary, mse, mae, city,
                train_start, train_end, test_start, test_end,
                seasonal_naive_mse, seasonal_naive_mae, sarima_mse, sarima_mae,
                sarima_order_p, sarima_order_d, sarima_order_q,
                sarima_seasonal_p, sarima_seasonal_d, sarima_seasonal_q,
                sarima_seasonal_period, model_path, code_version, data_hash,
                evaluation_folds, forecast_horizon_months, selected_model
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, (
                    adf_stat,
                    adf_pvalue,
                    adf_stat_diff,
                    adf_pvalue_diff,
                    order[0],
                    order[1],
                    order[2],
                    order[0],
                    order[1],
                    order[2],
                    forecast_steps,
                    model_summary,
                    mse,
                    mae,
                    city,
                    train_start,
                    train_end,
                    test_start,
                    test_end,
                    seasonal_naive_mse,
                    seasonal_naive_mae,
                    sarima_mse,
                    sarima_mae,
                    sarima_order[0],
                    sarima_order[1],
                    sarima_order[2],
                    sarima_seasonal_order[0],
                    sarima_seasonal_order[1],
                    sarima_seasonal_order[2],
                    sarima_seasonal_order[3],
                    model_path,
                    code_version,
                    data_hash,
                    evaluation_folds,
                    forecast_horizon_months,
                    selected_model,
                ))
        finally:
            conn.close()
