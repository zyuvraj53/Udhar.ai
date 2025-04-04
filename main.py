from data_preprocessing import load_and_clean_data

# ✅ Ensure the correct file path is passed
file_path = "Indian Banking Data.csv"

X_train, X_test, y_train, y_test, scaler = load_and_clean_data(file_path)
