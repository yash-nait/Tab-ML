from flask import ( Flask, render_template, request)
import os

# ML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = dir_path+"//static"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == "csv"

def predict(train_file, test_file):
    try:
        train = pd.read_csv(UPLOAD_FOLDER+"//"+train_file)
        X_train = train.drop(["target"], axis=1)
        y_train = train["target"].copy()
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

        rand_tree_reg = RandomForestRegressor()

        rand_tree_reg.fit(data_prepared, y_train)

        X_test = full_pipeline.fit_transform(pd.read_csv(UPLOAD_FOLDER+"//"+test_file))

        pred = rand_tree_reg.predict(X_test)

        pd.DataFrame(pred).to_csv(UPLOAD_FOLDER+"//prediction//prediction.csv", index=False)
        return True
    except Exception as e:
        print("err: "+str(e))
        return False

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/regression", methods=["GET", "POST"])
def regression():
    if request.method == "POST":
        train_file = request.files["train"]
        test_file = request.files["test"]
        if (train_file and test_file) and (allowed_file(train_file.filename) and allowed_file(test_file.filename)):
            train_location = os.path.join(
                UPLOAD_FOLDER,
                train_file.filename
            )
            test_location = os.path.join(
                UPLOAD_FOLDER,
                test_file.filename
            )
            train_file.save(train_location)
            test_file.save(test_location)

            pred = predict(train_file.filename, test_file.filename)
            
            if pred :
                return render_template("regressor.html", file_name="prediction.csv")
            else :
                return render_template("error.html")
        else:
            return render_template("error.html" , err="file err")
    return render_template("regression.html")

@app.route("/classification")
def classification():
    return render_template("classification.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)