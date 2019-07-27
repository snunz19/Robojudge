from ibm_watson import ToneAnalyzerV3
import json

text = "Hello, my name is Ronak Pai. Today, I will be presenting."
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='Iw1X1i5s-OK_c5RRmmWBVGRQyDwoqmtQ-NXvbQVBAcs7',
    url='https://gateway.watsonplatform.net/tone-analyzer/api'
)

tone_analysis = tone_analyzer.tone(
        {'text': text},
        content_type='application/json'
    ).get_result()

print(tone_analysis["document_tone"]["tones"][0]["tone_name"])

for sentence in tone_analysis["sentences_tone"]:
    print(sentence["text"])
    if len(sentence["tones"]) > 0:
        print(sentence["tones"][0]["tone_name"])

