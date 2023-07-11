import json
import gzip

import numpy as np
from datetime import datetime

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from summarizer import Summarizer


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from collections import defaultdict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
#from pylab import figure
from IPython.display import HTML
from matplotlib import animation
pyplot.rcParams['animation.ffmpeg_path'] = "C:\\FFmpeg\\bin\\ffmpeg.exe"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

website_dict = {
    "www.dailymail.co.uk": "dailymail",
    "www.reuters.com": "reuters",
    "www.channelnewsasia.com": "channelnewsasia",
    "www.cnn.com": "cnn",
    "in.reuters.com": "reuters",
    "www.seattletimes.com": "seattletimes",
    "www.express.co.uk": "express",
    "www.foxnews.com": "foxnews",
    "www.theguardian.com": "theguardian",
    "nationalpost.com": "nationalpost",
    "abcnews.go.com": "abcnews",
    "uk.reuters.com": "reuters",
    "sputniknews.com": "sputniknews",
    "www.westernjournal.com": "westernjournal",
    "www.nbcnews.com": "nbcnews",
    "www.npr.org": "npr",
    "www.washingtonpost.com": "washingtonpost",
    "indianexpress.com": "indianexpress",
    "www.bbc.com": "bbc",
    "www.metronews.ca": "metronews",
    "time.com": "time",
    "www.businessinsider.com": "businessinsider",
    "www.irishtimes.com": "irishtimes",
    "www.firstpost.com": "firstpost",
    "www.straitstimes.com": "straitstimes",
    "www.aljazeera.com": "aljazeera",
    "www.voanews.com": "voanews",
    "www.thenews.com.pk": "thenews",
    "www.startribune.com": "startribune",
    "www.yahoo.com": "yahoo",
    "www.ctvnews.ca": "ctvnews",
    "www.myplainview.com": "myplainview",
    "www.digitaljournal.com": "digitaljournal",
    "www.sfgate.com": "sfgate",
    "www.theglobeandmail.com": "theglobeandmail",
    "nation.com.pk": "nation",
    "www.dailysabah.com": "dailysabah",
    "www.nytimes.com": "nytimes",
    "www.bbc.co.uk": "bbc",
    "www.newindianexpress.com": "newindianexpress",
    "www.nytimes.com": "nytimes",
    "www.washingtontimes.com": "washingtontimes",
    "www.cnbc.com": "cnbc"
}


def l2_distance(A, B):
    return np.linalg.norm(A - B)


def compute_internal_distance_metric(embedings, function=l2_distance):
    internal_distance_metric = 0
    for i in range(len(embedings) - 1):
        for j in range(i + 1, len(embedings)):
            internal_distance_metric += function(embedings[i], embedings[j])

    # taking the average
    internal_distance_metric /= ((len(embedings) * (len(embedings) - 1)) / 2)
    return internal_distance_metric


def compute_distance_metric(embedings_a, embedings_b, function=l2_distance):
    distance_metric = 0
    for i in range(len(embedings_a)):
        for j in range(len(embedings_b)):
            distance_metric += function(embedings_a[i], embedings_b[j])

    distance_metric /= (len(embedings_a) * len(embedings_b)
                        )  # taking the average
    return distance_metric


def plot_comparison_of_events(embedings_A, embedings_B, labels_A, labels_B, title, creat_video=False):
    embedings = np.concatenate(
        (np.array(embedings_A), np.array(embedings_B)), axis=0)
    labels = np.concatenate((np.array(labels_A), np.array(labels_B)), axis=0)

    pca = PCA(n_components=3)
    embedings_SS = StandardScaler().fit_transform(embedings)
    embedings_SS_dim_reduced = pca.fit_transform(embedings_SS)

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(labels)):  # plot each point + it's index as text above
        if i < len(labels_A):
            ax.scatter(
                embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i, 1], embedings_SS_dim_reduced[i, 2], color='b')
        else:
            ax.scatter(
                embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i, 1], embedings_SS_dim_reduced[i, 2], color='r')
        ax.text(embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i, 1], embedings_SS_dim_reduced[i, 2],
                '%s' % (str(labels[i])), size=10, zorder=1, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

    pyplot.show()

    def animate(frame):
        ax.view_init(30, 20 + frame / 2)
        pyplot.pause(.001)
        return fig

    if creat_video:
        anim = animation.FuncAnimation(fig, animate, frames=720, interval=50)
        title = title.replace(" ", "_")
        anim.save(f"{title}_{labels_A[0]}_{labels_B[0]}_animation.mp4")


def compare_events(event_idx_a, event_idx_b, data, creat_video=False):

    event_a_articals = data[event_idx_a]["articles"]
    event_b_articals = data[event_idx_b]["articles"]

    print(f"event index {event_idx_a}, artical count: {len(event_a_articals)}")
    print(f"event index {event_idx_b}, artical count: {len(event_b_articals)}")
    print("")
    print(
        f"event index {event_idx_a}, summary: {data[event_idx_a]['summary']}")
    print(
        f"event index {event_idx_b}, summary: {data[event_idx_b]['summary']}")
    print("")

    event_a_CLSs = []
    event_a_avg_embs = []
    event_a_finetuned = []

    event_b_CLSs = []
    event_b_avg_embs = []
    event_b_finetuned = []

    for article in event_a_articals:
        temp_CLS = article["CLS"]
        temp_avg_emb = article["avg_embedings"]
        temp_finetuned = article["finetuned"]

        event_a_CLSs.append(np.array(temp_CLS))
        event_a_avg_embs.append(np.array(temp_avg_emb))
        event_a_finetuned.append(np.array(temp_finetuned))

    for article in event_b_articals:
        temp_CLS = article["CLS"]
        temp_avg_emb = article["avg_embedings"]
        temp_finetuned = article["finetuned"]

        event_b_CLSs.append(np.array(temp_CLS))
        event_b_avg_embs.append(np.array(temp_avg_emb))
        event_b_finetuned.append(np.array(temp_finetuned))

    # For Event A
    event_a_internal_CLS_l2 = compute_internal_distance_metric(
        event_a_CLSs, function=l2_distance)
    event_a_internal_avg_emb_l2 = compute_internal_distance_metric(
        event_a_avg_embs, function=l2_distance)
    event_a_internal_finetuned_l2 = compute_internal_distance_metric(
        event_a_finetuned, function=l2_distance)

    # for event B
    event_b_internal_CLS_l2 = compute_internal_distance_metric(
        event_b_CLSs, function=l2_distance)
    event_b_internal_avg_emb_l2 = compute_internal_distance_metric(
        event_b_avg_embs, function=l2_distance)
    event_b_internal_finetuned_l2 = compute_internal_distance_metric(
        event_b_finetuned, function=l2_distance)

    # for comperision
    event_a_and_b_CLS_l2 = compute_distance_metric(
        event_a_CLSs, event_b_CLSs, function=l2_distance)
    event_a_and_b_avg_emb_l2 = compute_distance_metric(
        event_a_avg_embs, event_b_avg_embs, function=l2_distance)
    event_a_and_b_finetuned_l2 = compute_distance_metric(
        event_a_finetuned, event_b_finetuned, function=l2_distance)

    print(
        f"event index {event_idx_a}, CLS internal l2 distane average: {event_a_internal_CLS_l2}")
    print(
        f"event index {event_idx_b}, CLS internal l2 distane average: {event_b_internal_CLS_l2}")
    print(
        f"CLS l2 distane average between the two events: {event_a_and_b_CLS_l2}")
    print("")
    print(
        f"event index {event_idx_a}, avg-embeding internal l2 distane average: {event_a_internal_avg_emb_l2}")
    print(
        f"event index {event_idx_b}, avg-embeding internal l2 distane average: {event_b_internal_avg_emb_l2}")
    print(
        f"avg-embeding l2 distane average between the two events: {event_a_and_b_avg_emb_l2}")
    print("")
    print(
        f"event index {event_idx_a}, finetunned embeding internal l2 distane average: {event_a_internal_finetuned_l2}")
    print(
        f"event index {event_idx_b}, finetunned embeding internal l2 distane average: {event_b_internal_finetuned_l2}")
    print(
        f"finetunned embeding l2 distane average between the two events: {event_a_and_b_finetuned_l2}")

    labels_A = [event_idx_a] * len(event_a_CLSs)
    labels_B = [event_idx_b] * len(event_b_CLSs)

    event_a_CLSs = np.array(event_a_CLSs)
    event_b_CLSs = np.array(event_b_CLSs)

    event_a_avg_embs = np.array(event_a_avg_embs)
    event_b_avg_embs = np.array(event_b_avg_embs)

    event_a_finetuned = np.array(event_a_finetuned)
    event_b_finetuned = np.array(event_b_finetuned)

    plot_comparison_of_events(event_a_CLSs, event_b_CLSs,
                              labels_A, labels_B, "CLS Embedding Plot", creat_video)
    plot_comparison_of_events(event_a_avg_embs, event_b_avg_embs,
                              labels_A, labels_B, "AVG Embedding Plot", creat_video)
    plot_comparison_of_events(event_a_finetuned, event_b_finetuned,
                              labels_A, labels_B, "Finetuned Embedding Plot", creat_video)


def plot_event(embedings, labels, title, creat_video=False):

    pca = PCA(n_components=3)
    embedings_SS = StandardScaler().fit_transform(embedings)
    embedings_SS_dim_reduced = pca.fit_transform(embedings_SS)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(labels)):  # plot each point + it's index as text above
        ax.scatter(embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i,
                                                                            1], embedings_SS_dim_reduced[i, 2], color='b')
        ax.text(embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i, 1], embedings_SS_dim_reduced[i, 2],
                '%s' % (str(i) + ", " + str(labels[i])), size=10, zorder=1, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

    pyplot.show()

    def animate(frame):
        ax.view_init(30, 20 + frame / 2)
        pyplot.pause(.001)
        return fig

    if creat_video:
        anim = animation.FuncAnimation(fig, animate, frames=720, interval=50)
        title = title.replace(" ", "_")
        anim.save(f"{title}_animation.mp4")


def summarize_text(text_content, summarizer_model):
    return summarizer_model(text_content, num_sentences=3)


def inspect_event(event_id, data, creat_video=False):

    event = data[event_id]
    articles_list = event["articles"]
    print(
        f"envent number {event_id}, number of articles: {len(articles_list)}")
    print(f"Summary: {event['summary']}")

    CLSs = []
    avg_embs = []
    finetuned = []
    for article in articles_list:
        temp_CLS = article["CLS"]
        temp_avg_emb = article["avg_embedings"]
        temp_finetuned = article["finetuned"]

        CLSs.append(np.array(temp_CLS))
        avg_embs.append(np.array(temp_avg_emb))
        finetuned.append(np.array(temp_finetuned))

    # Clustering algorithm here.

    # Determine the optimal number of clusters based on BIC
    n_components = np.arange(1, 10)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(
        CLSs) for n in n_components]
    bic = [model.bic(CLSs) for model in models]
    n_clusters = n_components[np.argmin(bic)]

    # Gaussian Mixture Clustering with optimal number of clusters
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)

    gmm.fit(CLSs)
    CLSs_cluster_labels = gmm.predict(CLSs)

    gmm.fit(avg_embs)
    avg_embs_cluster_labels = gmm.predict(avg_embs)

    gmm.fit(finetuned)
    finetuned_cluster_labels = gmm.predict(finetuned)

    # Store clusters
    CLSs_clusters = defaultdict(list)
    avg_embs_clusters = defaultdict(list)
    finetuned_clusters = defaultdict(list)

    for i, label in enumerate(CLSs_cluster_labels):
        # storing both the document and its CLS
        CLSs_clusters[label].append((articles_list[i], CLSs[i]))

    for i, label in enumerate(avg_embs_cluster_labels):
        # storing both the document and its avg embedding
        avg_embs_clusters[label].append((articles_list[i], avg_embs[i]))

    for i, label in enumerate(finetuned_cluster_labels):
        # storing both the document and its finetuned embedding
        finetuned_clusters[label].append((articles_list[i], finetuned[i]))

    # Summarization

    # Define the summarizer model
    summarizer_model = Summarizer()

    transition_phrases = ["Moreover, ", "Furthermore, ",
                          "In addition, ", "Similarly, ", "Also, "]
    for cluster_label, cluster_data in CLSs_clusters.items():
        print(f"\nCluster label: {cluster_label}")
        summary = ""
        for idx, (article, _) in enumerate(cluster_data):
            summarized_text = summarize_text(article['text'], summarizer_model)
            if idx > 0:
                summary += transition_phrases[idx %
                                              len(transition_phrases)] + summarized_text + ". "
            else:
                summary += summarized_text + ". "
        print(f"Summarized text: {summary}")

        # Generate a final summary for the whole cluster
        final_summary = summarize_text(summary, summarizer_model)
        print(f"Final Summary for cluster: {final_summary}")

    internal_CLS_l2 = compute_internal_distance_metric(
        CLSs, function=l2_distance)
    internal_avg_emb_l2 = compute_internal_distance_metric(
        avg_embs, function=l2_distance)
    internal_finetuned_l2 = compute_internal_distance_metric(
        finetuned, function=l2_distance)

    print(f"CLS internal l2 distane average: {internal_CLS_l2}")
    print(f"avg-embeding internal l2 distane average: {internal_avg_emb_l2}")
    print(
        f"finetuned embeding internal l2 distane average: {internal_finetuned_l2}")

    labels = []
    for article in articles_list:
        temp_website = article["website"]
        labels.append(website_dict[temp_website])

    plot_event(
        CLSs, labels, f"CLS Embedding event {event_id} Plot", creat_video=creat_video)
    plot_event(avg_embs, labels,
               f"AVG Embedding event {event_id} Plot", creat_video=creat_video)
    plot_event(finetuned, labels,
               f"Finetuned Embedding event {event_id} Plot", creat_video=creat_video)


def inspect_data(data):

    min_covarege_time_in_days = 100
    max_covarege_time_in_days = 1
    all_covarege_time_in_days = []

    for (i, event) in enumerate(data):

        articles_list = event["articles"]
        print(f"event number {i}, number of articles: {len(articles_list)}")
        print(f"Summary: {event['summary']}")

        CLSs = []
        avg_embs = []

        # taking the first articale with a date as the min and max date
        idx = 0
        while articles_list[idx]["time"] == None:
            idx += 1
        min_date = datetime.strptime(
            articles_list[idx]["time"][:19], "%Y-%m-%d %H:%M:%S")
        max_date = datetime.strptime(
            articles_list[idx]["time"][:19], "%Y-%m-%d %H:%M:%S")

        for article in articles_list:
            temp_CLS = article["CLS"]
            temp_avg_emb = article["avg_embedings"]
            CLSs.append(np.array(temp_CLS))
            avg_embs.append(np.array(temp_avg_emb))

            if article["time"] != None:
                date = datetime.strptime(
                    article["time"][:19], "%Y-%m-%d %H:%M:%S")
                if date < min_date:
                    min_date = date
                if date > max_date:
                    max_date = date

        internal_CLS_l2 = 0
        for i in range(len(CLSs) - 1):
            for j in range(i + 1, len(CLSs)):
                internal_CLS_l2 += l2_distance(CLSs[i], CLSs[j])

        # taking the average
        internal_CLS_l2 /= ((len(CLSs) * (len(CLSs) - 1)) / 2)

        internal_avg_emb_l2 = 0
        for i in range(len(avg_embs) - 1):
            for j in range(i + 1, len(avg_embs)):
                internal_avg_emb_l2 += l2_distance(avg_embs[i], avg_embs[j])

        # taking the average
        internal_avg_emb_l2 /= ((len(avg_embs) * (len(avg_embs) - 1)) / 2)

        covarege_time_in_days = (max_date - min_date).days
        all_covarege_time_in_days.append(covarege_time_in_days)

        if covarege_time_in_days < min_covarege_time_in_days:
            min_covarege_time_in_days = covarege_time_in_days
        if covarege_time_in_days > max_covarege_time_in_days:
            max_covarege_time_in_days = covarege_time_in_days

        print(f"CLS internal l2 distane average: {internal_CLS_l2}")
        print(
            f"avg-embeding internal l2 distane average: {internal_avg_emb_l2}")

        print(f"datetime of the first news: {min_date}")
        print(f"datetime of the last news: {max_date}")
        print(f"The event was covered for {covarege_time_in_days} days.\n")

    print(f"Min coverage time in days: {min_covarege_time_in_days}")
    print(f"Max coverage time in days: {max_covarege_time_in_days}")

    all_covarege_time_in_days_counts = dict()
    for i in all_covarege_time_in_days:
        all_covarege_time_in_days_counts[i] = all_covarege_time_in_days_counts.get(
            i, 0) + 1

    pyplot.hist(all_covarege_time_in_days, bins=1500)
    pyplot.show()
