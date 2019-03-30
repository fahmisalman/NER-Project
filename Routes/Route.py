from flask import Flask, jsonify, request
from Model import NER
import tensorflow as tf

app = Flask(__name__)


def model_check():
    if ner.p == '':
        ner.load_model()
    else:
        pass


def model_predict(s):
    model_check()
    return ner.predict(s)


@app.route('/', methods=['POST'])
def summary():
    result = ''
    if request.method == 'POST':
        sentence = request.form['sentence']
        graph = tf.get_default_graph()
        with graph.as_default():
            result = model_predict(sentence)
    return jsonify(result)


def main(host='0.0.0.0', port=8081):
    global ner, app
    ner = NER()
    app.run(host=host,
            debug=True,
            port=port)

