from flask import ( Flask, render_template, request)
import os

from sklearn import metrics

# ML
from models.regression import reg_predict
from models.classification import cls_predict

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = dir_path+"//static"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() == "csv"

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

            (pred, pos_err, score) = reg_predict(train_file.filename, test_file.filename, UPLOAD_FOLDER)
            
            if pred :
                return render_template("result.html", file_name="prediction.csv", score=score, metrics="Root Mean Square Error")
            else :
                return render_template("error.html", err=pos_err)
        else:
            return render_template("error.html" , err="file error")
    return render_template("regression.html")

@app.route("/classification", methods=["GET", "POST"])
def classification():
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

            (pred, pos_err, score) = cls_predict(train_file.filename, test_file.filename, UPLOAD_FOLDER)
            
            if pred :
                return render_template("result.html", file_name="prediction.csv", score=score, metrics="Accuracy")
            else :
                return render_template("error.html", err=pos_err)
        else:
            return render_template("error.html" , err="file error")
    return render_template("classification.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)