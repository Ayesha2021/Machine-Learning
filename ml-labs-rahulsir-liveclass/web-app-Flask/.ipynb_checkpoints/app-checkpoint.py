from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and transformers
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoder = data["encoder"]
scaler = data["scaler"]
num_cols = data["num_cols"]
cat_cols = data["cat_cols"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        location = request.form["location"]
        sqft = request.form["sqft"]
        beds = request.form["beds"]
        baths = request.form["baths"]
        year_built = request.form["year_built"]

        # Prepare input data
        input_df = pd.DataFrame([{
            "location": location,
            "sqft": float(sqft),
            "beds": int(beds),
            "baths": int(baths),
            "year_built": int(year_built)
        }])

        print(input_df)

        encoded_input = encoder.transform(input_df[cat_cols])
        scaled_input = scaler.transform(input_df[num_cols])

        final_input = pd.concat([
            pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_cols)),
            pd.DataFrame(scaled_input, columns=num_cols)
        ], axis=1)

        prediction = model.predict(final_input)[0]
        return render_template("predict.html", prediction=round(prediction, 2))

    return render_template("predict.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
