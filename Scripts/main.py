import json
import gzip

import numpy as np

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
#from pylab import figure
from IPython.display import HTML
from matplotlib import animation
from result_ploting import *
from BART_embeding_generation import *
from BART_finetuned_embeding_generation import *

pyplot.rcParams['animation.ffmpeg_path'] = "C:\\FFmpeg\\bin\\ffmpeg.exe"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def main():
    root_path = r"C:\Users\batua\PycharmProjects\csnlp-project"
    # Opening JSON file


    f = open(root_path + r"\Data\test_500_with_embedings+.json")

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # generate_embedings(5000, 5500, "test_500_with_embedings.json")

    # generate_embedings("test_500_with_embedings.json", "test_500_with_embedings+.json")

    # compare_events(0, 16, data, creat_video=True)

    # inspect_event(16, data, summarize=True, creat_video=False)

    compare_multiple_events(range(10), data, creat_video=False)

    # inspect_data(data)



# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    # generate_data()

    # read_data()

    # generate_embedings()

    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
