import numpy as np 
import pandas as pd 
from sklearn import preprocessing, model_selection

# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features

# no need to impute because preprocess all the data before splitting into train & test sets 

def load_data(name):
    """
    given a string filename, return a df
    """
    fn = "data/" + name + ".zip"
    return pd.read_csv(fn)


def exclude_cols(df, cols, exclude_first=False):
    """
    df: df 
    cols: list of string colnames 
    returns a df 
    exclude a list of cols from the df 
    """
    if exclude_first:
        df = df.drop(df.columns[0], axis=1)
    return df.drop(cols, axis=1)


def separate_target(df, col_name):
    """
    separate the target col from the feature cols 
    
    returns a list of targets and the df of features 
    """
    y = df.pop(col_name).values # mutates df 
    X = df
    return X, y
    
    
def encode_avo_avg_price(y):
    """
    binary classification task: 
    assign target values < median = 0
    assign target values > median = 1 
    """
    # print(y.shape)
    median_value = np.median(y)
    # print(median_value)
    enc_y = np.empty(y.shape)
    enc_y[y > median_value] = 1
    enc_y[y <= median_value] = 0
    # unique, counts = np.unique(enc_y, return_counts=True)
    # print(dict(zip(unique, counts)))
    return enc_y

def encode_odor(y):
    """
    binary classification task: 
    assign "t" = 1
    assign not "e" = 0
    """
    enc_y = np.empty(y.shape)
    enc_y[y == "n"] = 1
    enc_y[y != "n"] = 0
    # unique, counts = np.unique(enc_y, return_counts=True)
    # print(dict(zip(unique, counts)))
    return enc_y
    
    
def process_df(df):
    # one hot encode categorical data
    cat_df = df.select_dtypes(include=[object])
    ohe_arr, ohe_names = one_hot_encode(cat_df)
    # print("col names", ohe_names)
    # print("col names", ohe_names.shape)
    
    num_df = df.select_dtypes(include=[np.int64, np.float64])
    if num_df.columns.tolist():
        # normalize numeric data 
        norm_arr, norm_df_names = normalize(num_df)
        # print(norm_df_names)
        # print(norm_arr.shape)
        
        # add the cols together 
        X = np.hstack((ohe_arr, norm_arr))
        names = np.append(ohe_names, norm_df_names)
        
        # print(X.shape)
        # print(names)
        return X, names
        
    # print(ohe_arr.shape)
    # print(ohe_names)
    return ohe_arr, ohe_names
    

# one hot encode categorical
def one_hot_encode(cat_df):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    return ndarray, labels 
    """
    ohe = preprocessing.OneHotEncoder(drop="first")
    ohe_arr = ohe.fit_transform(cat_df).toarray()
    return np.array(ohe_arr), ohe.get_feature_names()


# normalize 
def normalize(num_df):
    """
    normalize numeric data 
    https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range
    """
    mms = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_arr = mms.fit_transform(num_df)
    return np.array(norm_arr), num_df.columns.tolist()


def get_avo_data(target_col="type"):
    """
    todo: maybe include month from date?
    avo data description:
        https://www.kaggle.com/neuromusic/avocado-prices
    """
    cols_to_exclude = ["region", "Date"]
    exclude_first_col = True
    df = load_data("avocado-prices")
    df = exclude_cols(df, cols_to_exclude, exclude_first_col)
    X, y = separate_target(df, target_col)
    if target_col == "AveragePrice":
        y = encode_avo_avg_price(y)
    X, x_names = process_df(X)
    return X, x_names, y
    
    
def get_mushroom_data(target_col="class"):
    """
    mushroom data description: 
        https://archive.ics.uci.edu/ml/datasets/Mushroom
        https://www.kaggle.com/uciml/mushroom-classification
    """
    df = load_data("mushroom-classification")
    cols_to_exclude = ["class", "gill-color", "cap-color", 
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "ring-type"]
    # ring type & class were too indicative of odor
    df = exclude_cols(df, cols_to_exclude)
    # remove "g" instances from "cap-surface"
    # df = df[df["cap-surface"] != "g"]
    X, y = separate_target(df, target_col)
    if target_col == "class":
        y = encode_m_edible(y)
    elif target_col == "stalk-shape":
        y = encode_stalk_shape(y)
    elif target_col == "odor":
        y = encode_odor(y)
    X, x_names = process_df(X)
    return X, x_names, y
    
    
def train_test_split(X, y, seed, test_size):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, 
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test


