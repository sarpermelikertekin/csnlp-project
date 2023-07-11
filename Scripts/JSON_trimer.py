import json

import numpy as np

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def trim_JSON():
    # Opening JSON file
    f = open('data_5000_with_BART_embedings.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # data = data[0:5000] # we will only look at some of the data

    # c = val_data[404]
    # summary = c['summary'] # human-written summary
    # articles = c['articles'] # cluster of articles
    # article_keys = articles[0].keys()


    for (i, event) in enumerate(data):
        print(f"event number {i}")
        articles_list = event["articles"]

        del [event["reference_urls"]]
        del [event["wiki_links"]]

        for article in articles_list:
            del (article["text"])
            del (article["avg_embedings"])
            if "events" in article:
                del (article["events"])
            del (article["id"])
            del (article["title"])
            del (article["url"])
            del (article["origin"])


    jsonString = json.dumps(data)
    jsonFile = open("data_5000_with_BART_embedings_CLS.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print("debug")

if __name__ == '__main__':

    trim_JSON()
