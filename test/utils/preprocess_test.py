import pandas as pd
from utils.preprocess import preprocess

# def preprocess(df):

test_df = pd.read_csv("../../data/test.csv")
df_preprocessed = preprocess(df = test_df)

print (df_preprocessed.isnull().sum() )