import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Keep only text + label
fake = fake[["text", "label"]]
true = true[["text", "label"]]

# Combine both datasets
data = pd.concat([fake, true], ignore_index=True)

# Remove empty rows
data.dropna(inplace=True)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Save final dataset
data.to_csv("data.csv", index=False)

print("✅ data.csv created successfully")