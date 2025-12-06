# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/sgpai/tourism-mlops/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
drop_cols = ['Unnamed: 0','CustomerID']
df.drop(drop_cols,axis=1,inplace=True)

df.drop_duplicates(inplace=True)

# Data correction
df['Gender'] = df['Gender'].replace('Fe Male','Female')
# Split into X (features) and y (target)
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']


# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rus = RandomUnderSampler(random_state=1, sampling_strategy = 1) # undersampling majority class to have the same counts as minorty class
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)


X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)
X_train_un.to_csv("X_train_un.csv",index=False)
y_train_un.to_csv("y_train_un.csv",index=False)


files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv","X_train_un.csv","y_train_un.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sgpai/tourism-mlops",
        repo_type="dataset",
    )
