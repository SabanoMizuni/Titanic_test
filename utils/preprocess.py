import pandas as pd

def preprocess(df):
    df = _sex_preprocess(df)
    df = _title_preprocess(df)
    df = _age_preprocess(df)
    df = _fare_preprocess(df)
    df = _embarked_preprocess(df)
    df = _familysize_preprocess(df)
    df_preprocessed = _drop_parameters(df)
    return df_preprocessed

def _sex_preprocess(df):
    df = df.replace("male", 0).replace("female", 1)
    return df

def _title_preprocess(df):
    # Preprocess "Title" by replacing the unfamiliar titles with more usual titles (i.e. Mr, Mrs, Miss and Master)
    df["Title"] = df["Name"].str.extract("([A-Za-z]+)\.", expand=True)  # pandas.Series.str.extract;

    title_dict = {"Sir": "Mr", "Major": "Mr", "Col": "Mr", "Rev": "Mr", "Dr": "Mr", "Capt": "Mr", "Jonkheer": "Mr",
                  "Don": "Mr", "Mme": "Mrs", "Lady": "Mrs", "Countess": "Mrs", "Ms": "Mrs", "Mlle": "Miss",
                  "Dona": "Mrs"}
    df.replace({"Title": title_dict}, inplace=True)
    df_2 = pd.get_dummies(df["Title"])
    df = pd.concat([df, df_2], axis=1)
    return df

def _age_preprocess(df):
    # Preprocess "Age" by substituting the missing value with the median value of other passangers of the same "Title" group.
    temp_ages = dict(df.groupby("Title")["Age"].median())
    df["median_age"] = df["Title"].apply(lambda x: temp_ages[x])
    df["Age"].fillna(df["median_age"], inplace=True, )
    del df["median_age"]
    return df

def _fare_preprocess(df):
    # Preprocess "Fare"
    temp_fares = dict(df.groupby("Title")["Fare"].median())
    df["median_fare"] = df["Title"].apply(lambda x: temp_fares[x])
    df["Fare"].fillna(df["median_fare"], inplace=True, )
    del df["median_fare"]
    del df["Title"] # Titles are no longer needed because they are string data.
    return df

def _embarked_preprocess(df):
    # Preprocess "Embarked"
    df.dropna(subset=["Embarked"], axis=0, inplace=True)
    df_2 = pd.get_dummies(df["Embarked"])
    del df["Embarked"]
    df = pd.concat([df, df_2], axis=1)
    return df

def _familysize_preprocess(df):
    # Preprocess "Parch" and "SibSp"
    df["FamilySize"] = df["Parch"] + df["SibSp"] + 1
    return df

def _drop_parameters(df):
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1) # drop 3 columns
    return df

