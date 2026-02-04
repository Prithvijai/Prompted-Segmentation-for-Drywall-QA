import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("logs/training_losses.xlsx")

plt.figure()
plt.plot(df["Epoch"], df["Train Loss"])
plt.plot(df["Epoch"], df["Val Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend(["Train Loss", "Validation Loss"])
plt.show()
