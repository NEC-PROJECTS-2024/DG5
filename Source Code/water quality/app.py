from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("water.pkl", "rb"))

@app.route('/')
def result():
    return render_template("front.html")

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]  # Convert to float instead of int
        final = [np.array(features)]
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)
        output_float = float(output)

        if output_float > 0.5:
            return render_template("front.html", pred="Water is safe to drink. Potability: {}".format(output_float))
        else:
            return render_template("front.html", pred="Water is unsafe. Do not drink. Potability: {}".format(output_float))
    else:
        return render_template("front.html", pred="Invalid request method.")

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=7000)
