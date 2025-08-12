from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = {
            'district': [request.form['district']],
            'neighborhood': [request.form['neighborhood']],
            'room': [int(request.form['room'])],
            'living room': [int(request.form['living_room'])],
            'area (m2)': [float(request.form['area'])],
            'age': [int(request.form['age'])],
            'floor': [int(request.form['floor'])]
        }
        df = pd.DataFrame(user_input)
        prediction = model.predict(df)[0]
        return render_template("index.html", prediction=round(prediction, 2))
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
