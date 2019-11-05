import pandas as pd
from sklearn.model_selection import train_test_split
from utils.preprocess import preprocess

def prep_dataset():
    """inport data"""
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    """preprocess the dataframe"""
    train_preprocessed = preprocess(df=train)
    test_preprocessed = preprocess(df=test)

    df_train, df_test = train_test_split(train_preprocessed, test_size=0.2, random_state=1)

    df_X_train = df_train.drop('Survived', axis=1)
    df_y_train = df_train['Survived']
    df_X_test = df_test.drop('Survived', axis=1)
    df_y_test = df_test['Survived']

    return df_X_train, df_y_train, df_X_test, df_y_test