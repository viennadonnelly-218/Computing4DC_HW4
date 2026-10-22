# Finally, you are going to perform some data analysis on the data from “sample_diabetes_mellitus_data.csv”. 
# Imagine you are in a project that tries to predict whether some patients have diabetes mellitus. In order
# to do so, you create a library that will be used for your team in order to perform the analysis and make
# predictions. Since you want your colleagues to make the code as understandable as possible (and earn
# the highest grade possible), follow the best practices learnt during the lessons. Try to modularise the
# steps into functions (you can potentially merge g and h) and to separate the functions into files/folders
# in a way that makes the most sense to the team. For the purpose of this exercise, you are going to follow the next steps:
# a. Load the data.
# b. Split the data between train and test. (you can use train_test_split from sklearn or any other way)
# c. Remove those rows that contain NaN values in the columns: age, gender, ethnicity.
# d. Fill NaN with the mean value of the column in the columns: height, weight.
# e. Generate dummies for ethnicity column (One hot encoding).
# f. Create a binary variable for gender M/F.
# g. Train a model (for instance LogisticRegression or RandomForestClassifier from sklearn) in the train data.
# Use as features the columns: ‘age’, ‘height’, ‘weight’, ‘aids’, ‘cirrhosis’, ‘hepatic_failure’,‘immunosuppression’,
# ‘leukemia’, ‘lymphoma’, ‘solid_tumor_with_metastasis’. Use as target the column: ‘diabetes_mellitus’
# h. Predict the targets for both the train and test sets and add the prediction as a new column (use predict_proba
# from the model to get the predicted probabilities) name the new column something like predictions.
# i. Compute the train and test roc_auc metric using roc_auc_score from sklearn.

import pandas as pd
from sklearn.model_selection import train_test_split

# a. load data
df = pd.read_csv("data/sample_diabetes_mellitus_data.csv")

# print(df.head)
# print(df.info)
# print(df.columns)
# print(df.isnull().sum())


# b. Train/test split
train, test = train_test_split(df, test_size=0.2, random_state=42)


# c. remove NaN in the columns: age, gender, ethnicity
train = train.dropna(subset=['age', 'gender', 'ethnicity'])
# print(train.isnull().sum())
test = test.dropna(subset=['age', 'gender', 'ethnicity'])
# print(test.isnull().sum())

# d. Fill NaN with the mean value of the column in the columns: height, weight.

def fill_missing_with_mean(df, columns):
    """Replace NaN values in the given columns with the column mean"""
    df = df.copy()
    df[columns] = df[columns].fillna(df[columns].mean())
    return df

train = fill_missing_with_mean(train, ["height", "weight"])
test = fill_missing_with_mean(test, ["height", "weight"])
print(train[["height", "weight"]].isnull().sum())

# e. Generate dummies for ethnicity column (One hot encoding).

def add_one_hot_dummies(df, columns):
    """Add one-hot encoded columns for a categorical column."""
    dummies = pd.get_dummies(df[columns], prefix=columns, dtype=int)
    return pd.concat([df, dummies], axis=1)

train = add_one_hot_dummies(train, "ethnicity")
test = add_one_hot_dummies(test, "ethnicity")
print(train.filter(like="ethnicity").head())

# f. Create a binary variable for gender M/F.

def encode_binary(df, columns, positive_value, new_column):
    """ Create a 0/1 column: 1 if df[columns] == positive_value."""
    df = df.copy()
    df[new_column] = (df[columns] == positive_value).astype(int)
    return df

train = encode_binary(train, "gender", "M", "gender_male")
test = encode_binary(test, "gender", "M", "gender_male")
print(train[["gender", "gender_male"]].head())