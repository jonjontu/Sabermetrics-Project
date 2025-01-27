from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

dict_dat = pd.read_csv("/Users/gamerboyj/data-dictionary.csv")
train_dat = pd.read_csv("/Users/gamerboyj/data-train.csv")
test_dat = pd.read_csv("/Users/gamerboyj/data-test.csv")
preds_dat = pd.read_csv("/Users/gamerboyj/test_probs.csv")
features_dat = joblib.load("/Users/gamerboyj/feature_pairs.pkl")
model_info = joblib.load("/Users/gamerboyj/model_info.pkl")

@app.route("/")
def home():
    return render_template("home.html", 
                           train_col_names=train_dat.columns.tolist(), 
                           test_col_names=test_dat.columns.tolist(),
                           train_head_dat=train_dat.head(10).to_dict(orient="records"),
                           test_head_dat=test_dat.head(10).to_dict(orient="records"))
                           
@app.route("/features")
def features():
    return render_template("features.html",
                           dict_col_names=dict_dat.columns.tolist(), 
                           dict_dat=dict_dat.to_dict(orient="records"))

@app.route("/results")
def results():
    return render_template("results.html",
                           preds_col_names=preds_dat.columns.tolist(),
                           preds_head_dat=preds_dat.head(10).to_dict(orient="records"),
                           model_info=model_info,
                           features_dat=features_dat)

if __name__ == "__main__":
    app.run(debug=True)