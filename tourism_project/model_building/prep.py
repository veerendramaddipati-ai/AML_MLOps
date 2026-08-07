
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATA_PATH)

# Drop 'Unnamed: 0' column if it exists (often created when saving/loading CSVs with default index)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove unnecessary identifier column
df.drop(columns=["CustomerID"], inplace=True)

# Save complete dataset structure
target = "ProdTaken"

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.20,
    random_state=42
)

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)

y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

print("Train Test Split Saved Successfully")
