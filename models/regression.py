import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

#PREPROCESSING
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

#METRICS
from sklearn.metrics import mean_squared_error

#MODELS
from sklearn.ensemble import RandomForestRegressor

def reg_predict(train_file, test_file, UPLOAD_FOLDER):
    try:
        train = pd.read_csv(UPLOAD_FOLDER+"//"+train_file)
        train, val = train_test_split(train, test_size=0.2, random_state=42)
        X_train = pd.DataFrame(train).drop(["target"], axis=1)
        y_train = pd.DataFrame(train)["target"].copy()
        num = []
        cat = []
        for att in X_train.columns.values:
            if X_train[att].dtype == np.dtype('O'):
                cat.append(att)
            else :
                num.append(att)
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scalar", StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer_cat", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder())
        ])

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num),
            ("cat", cat_pipeline, cat)
        ])

        data_prepared = full_pipeline.fit_transform(X_train)

        rand_tree_reg = RandomForestRegressor(random_state=42)

        rand_tree_reg.fit(data_prepared, y_train)

        #validation score
        X_val = pd.DataFrame(val).drop(["target"], axis=1)
        y_val = pd.DataFrame(val)["target"].copy()
        X_val = full_pipeline.fit_transform(X_val)
        rmse = np.sqrt(mean_squared_error(rand_tree_reg.predict(X_val), y_val))

        X_test = full_pipeline.fit_transform(pd.read_csv(UPLOAD_FOLDER+"//"+test_file))

        pred = rand_tree_reg.predict(X_test)

        pd.DataFrame(pred).to_csv(UPLOAD_FOLDER+"//prediction//prediction.csv", header=['prediction'], index=False)
        return (True, "", rmse)
    except Exception as e:
        print("err: "+str(e))
        return (False, str(e), np.nan)