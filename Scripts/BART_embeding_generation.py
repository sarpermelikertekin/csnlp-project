import json

import numpy as np

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def generate_embedings():
    # Opening JSON file
    f = open('data.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    data = data[0:5000] # we will only look at some of the data

    # c = val_data[404]
    # summary = c['summary'] # human-written summary
    # articles = c['articles'] # cluster of articles
    # article_keys = articles[0].keys()

    # Downloading the BART model and picking just the encoder part of it.
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    encoder = model.model.encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder = encoder.to(device)

    def text_to_embedings(text):  # the function that generates the embedings for text using the BART encoder.

        text_input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024)['input_ids'].to(
            device)
        encodings = encoder(text_input_ids)

        encodings_array = np.squeeze(encodings.last_hidden_state.cpu().numpy())

        CLS = encodings_array[0, :]
        embedding_avg = np.sum(encodings_array[1:, :], axis=0) / (encodings_array.shape[1] - 1)

        return CLS, embedding_avg

    # adding embedings
    new_data = []
    for (i, event) in enumerate(data):
        print(f"event number {i}")
        articles_list = event["articles"]
        new_articles_list = []
        for article in articles_list:

            url = article["url"]

            website = (url.split("//")[1]).split("/")[0]

            CLS, avg_embedings = text_to_embedings(article["text"])

            article["website"] = website
            article["CLS"] = CLS.tolist()
            article["avg_embedings"] = avg_embedings.tolist()

            new_articles_list.append(article)

        event["articles"] = new_articles_list
        new_data.append(event)
    data = new_data



    jsonString = json.dumps(data)
    jsonFile = open("data_5000_with_BART_embedings.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print("debug")

if __name__ == '__main__':

    generate_embedings()

