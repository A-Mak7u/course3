import pandas as pd

try:
 df = pd.read_csv("lstm3_1.csv")
 print("Current best metrics (sorted by MSE):")
 print(df.head(1)[["Batch Size", "Activation", "Hidden Size", "Num Layers", "Dropout", "Learning Rate", "Epochs", "MSE", "R2"]])
except FileNotFoundError:
 print("File lstm3_1.csv not found. Make sure the training script is running.")
except pd.errors.EmptyDataError:
 print("File lstm3_1.csv is empty. No results saved yet.")