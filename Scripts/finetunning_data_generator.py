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

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


class Random_atlas():

    def __init__(self, size=10000):
        self.size = size
        self.index = 0
        self.atlas = np.random.rand(self.size)

    def sample_atlas(self):
        if self.index >= self.size:
            self.index = 0
            self.atlas = np.random.rand(self.size)
        res = self.atlas[self.index]
        self.index += 1
        return res

    def int_in_range(self, range, blacklist):
        min, max = range
        dif = max - min

        # giving an error for to large black list.
        if len(blacklist) >= dif:
            # logging.error("The blacklist is covering the range can't get a random number")
            return 0, False # False is to indicate that the function can not return a random int

        # we sample a random integer in the given range
        random_num = self.sample_atlas()
        res = math.floor(random_num * dif) + min

        # we keep on re-sampling random integers until we get one not in the blacklist
        while res in blacklist:
            random_num = self.sample_atlas()
            res = math.floor(random_num * dif) + min

        return res, True # True is to indicate that the function returned a random int


#
def generate_finetunning_data_saver(number_of_samples):
    # Opening JSON file
    f = open('mini_data_500_with_embedings.json')
    events = json.load(f)

    # This give us the randomness
    random_atlas = Random_atlas(size=20000)

    # this will be the JSON we will save as the date.
    finetunning_data = []

    num_events = len(events)

    for (event_idx, event) in enumerate(events):
        print(f"event number {event_idx}")
        articles_list = event["articles"]

        num_articles = len(articles_list)

        # black list so that the same sample is not added twice
        inner_blacklist_dict = {i: [i] for i in range(num_articles)}

        for (article_idx, article) in enumerate(articles_list):

            # first we add same event samples
            for _ in range(number_of_samples):

                # get list of articles we should not sample from
                temp_blacklist = inner_blacklist_dict[article_idx]

                # get the random sample index, or learn that there are no random index to get
                other_article_index, found_random_int = random_atlas.int_in_range(range=(0, num_articles), blacklist=temp_blacklist)

                if found_random_int:

                    # update the blacklist
                    inner_blacklist_dict[article_idx].append(other_article_index)
                    inner_blacklist_dict[other_article_index].append(article_idx)

                    # add the new sample and its mirror
                    finetunning_data.append({"A_event_idx": event_idx,
                                             "A_article_idx": article_idx,
                                             "A_emb_CLS": article["CLS"],
                                             "A_emb_AVG": article["avg_embedings"],
                                             "B_event_idx": event_idx,
                                             "B_article_idx": other_article_index,
                                             "B_emb_CLS": articles_list[other_article_index]["CLS"],
                                             "B_emb_AVG": articles_list[other_article_index]["avg_embedings"],
                                             "lable": 1
                                             })

                    finetunning_data.append({"A_event_idx": event_idx,
                                             "A_article_idx": other_article_index,
                                             "A_emb_CLS": articles_list[other_article_index]["CLS"],
                                             "A_emb_AVG": articles_list[other_article_index]["avg_embedings"],
                                             "B_event_idx": event_idx,
                                             "B_article_idx": article_idx,
                                             "B_emb_CLS": article["CLS"],
                                             "B_emb_AVG": article["avg_embedings"],
                                             "lable": 1
                                             })

            # second we add different event samples
            for _ in range(number_of_samples):
                # in this step we can not, fail to get a random int, so we ignore the true/false.

                # get the random sample event index, we must not sample from the same event so its is black listed
                other_event_index, _ = random_atlas.int_in_range(range=(0, num_events), blacklist=[event_idx])

                # get the random sample article index, we ignore the posibility of sampleing the same article here as is quite small
                other_event_num_articles = len(events[other_event_index]["articles"])
                other_article_index, _ = random_atlas.int_in_range(range=(0, other_event_num_articles), blacklist=[])

                # add the new sample and its mirror
                finetunning_data.append({"A_event_idx": event_idx,
                                         "A_article_idx": article_idx,
                                         "A_emb_CLS": article["CLS"],
                                         "A_emb_AVG": article["avg_embedings"],
                                         "B_event_idx": other_event_index,
                                         "B_article_idx": other_article_index,
                                         "B_emb_CLS": events[other_event_index]["articles"][other_article_index]["CLS"],
                                         "B_emb_AVG": events[other_event_index]["articles"][other_article_index]["avg_embedings"],
                                         "lable": 0
                                         })

                finetunning_data.append({"A_event_idx": other_event_index,
                                         "A_article_idx": other_article_index,
                                         "A_emb_CLS": events[other_event_index]["articles"][other_article_index]["CLS"],
                                         "A_emb_AVG": events[other_event_index]["articles"][other_article_index]["avg_embedings"],
                                         "B_event_idx": event_idx,
                                         "B_article_idx": article_idx,
                                         "B_emb_CLS": article["CLS"],
                                         "B_emb_AVG": article["avg_embedings"],
                                         "lable": 0
                                         })

    print("staring to write the JSON")

    start = time.time()

    # jsonString = json.dumps(finetunning_data)
    # jsonFile = open("finetunning_mini_data_500_BART_embedings.json", "w")
    # jsonFile.write(jsonString)
    with open("finetunning_mini_data_500_BART_embedings.json", 'w') as f:
        f.write('[')

        for obj in finetunning_data[:-1]:
            json.dump(obj, f)
            f.write(',')

        json.dump(finetunning_data[-1], f)
        f.write(']')

    f.close()

    end = time.time()
    print(f"JSON wrting took: {end - start}")

    print("debug")


def generate_finetunning_data(events, number_of_samples, emb_type="CLS"):
    # Opening JSON file
    events = events

    # This give us the randomness
    random_atlas = Random_atlas(size=20000)

    # this will be the JSON we will save as the date.
    finetunning_data_0 = []
    finetunning_data_1 = []
    finetunning_lables = []
    debuging_lookup = []

    num_events = len(events)

    for (event_idx, event) in enumerate(events):
        print(f"event number {event_idx}")
        articles_list = event["articles"]

        num_articles = len(articles_list)

        # black list so that the same sample is not added twice
        inner_blacklist_dict = {i: [i] for i in range(num_articles)}

        for (article_idx, article) in enumerate(articles_list):

            # first we add same event samples
            for _ in range(number_of_samples):

                # get list of articles we should not sample from
                temp_blacklist = inner_blacklist_dict[article_idx]

                # get the random sample index, or learn that there are no random index to get
                other_article_index, found_random_int = random_atlas.int_in_range(range=(0, num_articles), blacklist=temp_blacklist)

                if found_random_int:

                    # update the blacklist
                    inner_blacklist_dict[article_idx].append(other_article_index)
                    inner_blacklist_dict[other_article_index].append(article_idx)

                    # add the new sample and its mirror
                    if emb_type=="CLS":
                        finetunning_data_0.append(article["CLS"])
                        finetunning_data_1.append(articles_list[other_article_index]["CLS"])
                        finetunning_lables.append(1)
                        debuging_lookup.append((event_idx, article_idx, event_idx, other_article_index, 1))

                        finetunning_data_0.append(articles_list[other_article_index]["CLS"])
                        finetunning_data_1.append(article["CLS"])
                        finetunning_lables.append(1)
                        debuging_lookup.append((event_idx, other_article_index, event_idx, article_idx, 1))
                    else:
                        logging.error(f"Only embedding types are 'CLS','AVG', and a different embedding type is given.")

            # second we add different event samples
            for _ in range(number_of_samples):
                # in this step we can not, fail to get a random int, so we ignore the true/false.

                # get the random sample event index, we must not sample from the same event so its is black listed
                other_event_index, _ = random_atlas.int_in_range(range=(0, num_events), blacklist=[event_idx])

                # get the random sample article index, we ignore the posibility of sampleing the same article here as is quite small
                other_event_num_articles = len(events[other_event_index]["articles"])
                other_article_index, _ = random_atlas.int_in_range(range=(0, other_event_num_articles), blacklist=[])

                # add the new sample and its mirror
                if emb_type == "CLS":
                    finetunning_data_0.append(article["CLS"])
                    finetunning_data_1.append(events[other_event_index]["articles"][other_article_index]["CLS"])

                    finetunning_lables.append(0)
                    debuging_lookup.append((event_idx, article_idx, other_event_index, other_article_index, 0))

                    finetunning_data_0.append(events[other_event_index]["articles"][other_article_index]["CLS"])
                    finetunning_data_1.append(article["CLS"])

                    finetunning_lables.append(0)
                    debuging_lookup.append((other_event_index, other_article_index, event_idx, article_idx, 0))

                else:
                    logging.error(f"Only embedding types are 'CLS','AVG', and a different embedding type is given.")

    return finetunning_data_0, finetunning_data_1, finetunning_lables, debuging_lookup
    print("debug")


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

def create_loader_from_np(X_1, X_2, y=None, train=True, batch_size=32, shuffle=True, num_workers=1):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X_1).type(torch.float),
                                torch.from_numpy(X_2).type(torch.float),
                                torch.from_numpy(y).type(torch.long))
        loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)

    else:
        dataset = TensorDataset(torch.from_numpy(X_1).type(torch.float),
                                torch.from_numpy(X_2).type(torch.float))
        loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)
    return loader

def create_loader_from_list(X_1, X_2, y=None, train=True, batch_size=32, shuffle=True, num_workers=1):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.Tensor(X_1).type(torch.float),
                                torch.Tensor(X_2).type(torch.float),
                                torch.Tensor(y).type(torch.long))
        loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)

    else:
        dataset = TensorDataset(torch.Tensor(X_1).type(torch.float),
                                torch.Tensor(X_2).type(torch.float))
        loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)
    return loader

def finetune_model(encoder, net, train_loader, val_loader):

    train_loader = train_loader
    val_loader = val_loader


    encoder = encoder
    model = net

    model.to(device)

    n_epochs = 100


    criterion = nn.BCELoss()

    # create your optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    val_losses = []
    train_losses = []

    for epoch in range(n_epochs):
        model.train()
        loss_to_print = 0.0
        e_loss = 0.0
        predictions = []
        y_true = []
        for i, [X_1, X_2, y] in enumerate(train_loader):
            X_1 = X_1.to(device)
            X_2 = X_2.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            preds = model(X_1, X_2)
            loss = criterion(torch.reshape(preds, (-1,)), y.to(torch.float32))
            loss.backward()
            optimizer.step()
            e_loss += loss.item()
            loss_to_print += loss.item()
            if i % 1000 == 999:
                print(f"[{epoch + 1}, {i + 1:5d}], loss: {loss_to_print / 1000:.3f}")
                loss_to_print = 0.0
        print(f"[Epoch {epoch + 1:2d}] Epoch Loss: {e_loss / len(train_loader):.5f}", end=" | ")
        train_losses.append(e_loss / len(train_loader))
        model.eval()
        val_loss_p = 0

        with torch.no_grad():  # We don't need to compute gradients for testing
            for [x_val_1, x_val_2, y_val] in val_loader:
                x_val_1 = x_val_1.to(device)
                x_val_2 = x_val_2.to(device)
                y_val = y_val.to(device)
                y_hat = model(x_val_1, x_val_2)
                val_loss = criterion(torch.reshape(y_hat, (-1,)), y_val.to(torch.float32))
                val_loss_p += val_loss.item()
                predicted = y_hat.cpu().numpy()
                y_batch = y_val.cpu().numpy()

                # Rounding the predictions to 0 or 1
                predicted[predicted >= 0.5] = 1
                predicted[predicted < 0.5] = 0
                predictions.append(predicted)
                y_true.extend(y_batch)

        predictions = np.vstack(predictions)
        val_loss_p = val_loss_p / len(val_loader)
        acc = accuracy_score(y_true, predictions)
        print(f"Val Loss: {val_loss_p:.5f}, Val Accuracy: {acc:.5f}")
        val_losses.append(val_loss_p)

    torch.save(encoder, f"./encoder.pt")
    torch.save(model, f"./model.pt")

    return train_losses, val_losses

if __name__ == '__main__':

    f = open('mini_data_500_with_embedings.json')
    events = json.load(f)
    events_train = events[:450]
    events_val = events[450:]


    finetunning_data_train_X_0, finetunning_data_train_X_1, finetunning_data_train_y, debuging_lookup_tr = generate_finetunning_data(events_train, number_of_samples=5, emb_type="CLS")
    finetunning_data_val_X_0, finetunning_data_val_X_1, finetunning_data_val_y, debuging_lookup_val = generate_finetunning_data(events_val, number_of_samples=5, emb_type="CLS")

    encoder = Encoder()
    net = Net(encoder)


    train_loader = create_loader_from_list(finetunning_data_train_X_0, finetunning_data_train_X_1, finetunning_data_train_y, batch_size=256, shuffle=True)
    val_loader = create_loader_from_list(finetunning_data_val_X_0, finetunning_data_val_X_1, finetunning_data_val_y, batch_size=256, shuffle=False)

    train_losses, val_losses = finetune_model(encoder, net, train_loader, val_loader)

    # t = range(1, len(train_losses)+1)
    # pyplot.plot(t, train_losses, 'b')  # plotting t, b separately
    # pyplot.plot(t, val_losses, 'r')  # plotting t, c separately
    # pyplot.show()

    from matplotlib import pyplot

























