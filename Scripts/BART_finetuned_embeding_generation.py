import json
import logging

import numpy as np

import math


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# from matplotlib import pyplot


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Encoder(nn.Module):
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # and then used to extract features from the training and test data.

        # 1024 ==> 1024
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU()
        )

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # defined in the constructor.
        encoded = self.encoder(x)

        return encoded


class Net(nn.Module):
    def __init__(self, encoder):
        """
        The constructor of the model.
        """
        super().__init__()
        # and then used to extract features from the training and test data.

        # 1024 ==> 1024
        # self.encoder_1 = encoder
        # self.encoder_2 = Encoder()
        self.encoder = encoder

        # 2048 ==> 1
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        )

    def forward(self, x_1, x_2):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # defined in the constructor.
        encoded_1 = self.encoder(x_1)
        encoded_2 = self.encoder(x_2)
        encoded = torch.cat((encoded_1, encoded_2), 1)
        res = self.fc(encoded)
        res = F.sigmoid(res)
        return res


def add_embedings(input_data_name, save_name, encoder_name, new_embeding_name):
    # Opening JSON file
    root_path = r"C:\Users\batua\PycharmProjects\csnlp-project"


    f = open(root_path + "\\Data\\" + input_data_name)

    # returns JSON object as
    # a dictionary
    data = json.load(f)



    # c = val_data[404]
    # summary = c['summary'] # human-written summary
    # articles = c['articles'] # cluster of articles
    # article_keys = articles[0].keys()


    encoder = torch.load(root_path + "\\Scripts\\" + encoder_name)
    # encoder = encoder.to(device)

    encoder.cpu()

    # adding embedings
    for (i, event) in enumerate(data):
        print(f"event number {i}")
        articles_list = event["articles"]
        for article in articles_list:

            CLS = article["CLS"]

            encodings = encoder(torch.Tensor(CLS).type(torch.float)).detach().numpy().tolist()

            article[new_embeding_name] = encodings




    jsonString = json.dumps(data)
    jsonFile = open(root_path + "\\Data\\" + save_name, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print("debug")

if __name__ == '__main__':

    add_embedings()

