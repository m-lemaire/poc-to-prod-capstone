from flask import Flask, request, render_template
from run import TextPredictionModel

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('app.html')


@app.route("/", methods=['POST'])
def get_prediction():
    model = TextPredictionModel.from_artefacts("D:\COURS\EPF\5A\Poc to Prod\poc-to-prod-capstone\train\data\artefacts\2023-01-03-10-19-58")
    text = request.form['text']
    prediction = model.predict(text)
    return str(prediction)


if __name__ == '__main__':
    app.run()
    debug = True