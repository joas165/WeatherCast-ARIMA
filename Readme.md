# 🌍 Weather Forecasting with ARIMA 📊

This project performs time series analysis and forecasting using ARIMA models on global temperature data.

## 🚀 Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/weather_project.git
   cd weather_project
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Download the dataset:**
   - The dataset (`GlobalLandTemperaturesByMajorCity.csv`) originates from [Berkeley Earth](http://berkeleyearth.org/data/) and is also referenced in the [Climate Repository](https://github.com/gindeleo/climate/tree/main/data). Please refer to their licensing terms before use.
   - Place it in the `data/` folder.

## ▶️ Running the Project

```sh
python weather_project.py
```

## 📝 Notes
- 📂 The SQLite database (`weatherDatabase.db`) is created automatically inside the `data/` directory.
- 🛑 The `.gitignore` file ensures database and CSV files are not committed.
- 📊 The raw data originates from [Berkeley Earth](http://berkeleyearth.org/data/) and is also referenced in the [Climate Repository](https://github.com/gindeleo/climate/tree/main/data).

## 📜 License
This project is licensed under the **MIT License**.


