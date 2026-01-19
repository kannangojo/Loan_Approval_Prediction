from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

model=joblib.load("loan_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    income = float(request.form["income"])
    credit_score = float(request.form["credit_score"])

    input1= np.array([[income, credit_score]])
    prediction = model.predict(input1)

    result = " Loan Approved" if prediction[0] == 1 else " Loan Rejected"

    return render_template("index.html", prediction=result)

if __name__=="__main__":
    app.run(debug=True)