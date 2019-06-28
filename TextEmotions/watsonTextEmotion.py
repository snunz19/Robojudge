from ibm_watson import ToneAnalyzerV3
from flask import Flask, render_template, request, redirect, Response

app = Flask(__name__)

text = ""

tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='Iw1X1i5s-OK_c5RRmmWBVGRQyDwoqmtQ-NXvbQVBAcs7',
    url='https://gateway.watsonplatform.net/tone-analyzer/api'
)


@app.route('/')
def start():
    return render_template("index.html")


@app.route('/recieve', methods = ['POST', 'GET'])
def getText():
    data = request.get_json(force=True)
    text = str(data[0])
    print(text)
    '''
    tone_analysis = tone_analyzer.tone(
        {'text': text},
        content_type='application/json'
    ).get_result()
    '''
    render_template("index.html")
    return "success"


#Run the app!
if __name__ == "__main__":
    app.run(debug = False)

