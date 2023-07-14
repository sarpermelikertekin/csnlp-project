import json
import gzip

import numpy as np
from datetime import datetime

import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score


from collections import defaultdict

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
#from pylab import figure
from IPython.display import HTML
from matplotlib import animation
pyplot.rcParams['animation.ffmpeg_path'] = "C:\\FFmpeg\\bin\\ffmpeg.exe"
root_path = r"C:\Users\batua\PycharmProjects\csnlp-project"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

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
        anim.save(root_path + "\\Plot_animations\\" + f"{title}_{labels_A[0]}_{labels_B[0]}_animation.mp4")


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


def plot_event(embedings, labels, clusters, title, creat_video=False):

    cluster_colors = ["b", "r", "g", "m", "c", "b", "y"]
    legend_labels = [pyplot.Line2D([0], [0], marker='o', color='b', label='Cluster 0', markersize=10),
                     pyplot.Line2D([0], [0], marker='o', color='r', label='Cluster 1', markersize=10),
                     pyplot.Line2D([0], [0], marker='o', color='g', label='Cluster 2', markersize=10),
                     pyplot.Line2D([0], [0], marker='o', color='m', label='Cluster 3', markersize=10),
                     pyplot.Line2D([0], [0], marker='o', color='c', label='Cluster 4', markersize=10),
                     pyplot.Line2D([0], [0], marker='o', color='b', label='Cluster 5', markersize=10),
                     pyplot.Line2D([0], [0], marker='o', color='y', label='Cluster 6', markersize=10)]

    pca = PCA(n_components=3)
    embedings_SS = StandardScaler().fit_transform(embedings)
    embedings_SS_dim_reduced = pca.fit_transform(embedings_SS)

    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(labels)):  # plot each point + it's index as text above
        ax.scatter(embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i, 1], embedings_SS_dim_reduced[i, 2],
                   color=cluster_colors[clusters[i]], s=50)
        # ax.text(embedings_SS_dim_reduced[i, 0], embedings_SS_dim_reduced[i, 1], embedings_SS_dim_reduced[i, 2],
        #         '%s' % (str(i) + ", " + str(labels[i])), size=10, zorder=1, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    num_clusters = max(clusters) + 1
    ax.legend(handles=legend_labels[:num_clusters], loc='lower left', title='Legend')

    ax.text2D(0.05, 0.95, title, transform=ax.transAxes)

    pyplot.show()

    def animate(frame):
        ax.view_init(30, 20 + frame / 2)
        pyplot.pause(.001)
        return fig

    if creat_video:
        anim = animation.FuncAnimation(fig, animate, frames=720, interval=50)
        title = title.replace(" ", "_")
        anim.save(root_path + "\\Plot_animations\\" + f"{title}_animation.mp4")


def summarize_text(text_content, summarizer_model, num_sentences=3):
    return summarizer_model(text_content, num_sentences=num_sentences)


def inspect_event(event_id, data, summarize=False, creat_video=False):

    event = data[event_id]
    articles_list = event["articles"]
    print(f"envent number {event_id}, number of articles: {len(articles_list)}")
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
    explained_var = 0.999
    def reduce_dim_with_PCA(data, explained_var):

        pca = PCA()
        data_SS = StandardScaler().fit_transform(np.array(data))
        pca.fit(data_SS)

        for idx, i in enumerate(np.cumsum(pca.explained_variance_ratio_)):
            if i > explained_var:
                reduced_dim = idx + 1
                break

        pca = PCA(n_components=reduced_dim)

        data_SS_dim_reduced = pca.fit_transform(data_SS)
        pca.fit(data_SS)

        return data_SS_dim_reduced

    def cluster_GMM(embedings_list, explained_var):
        embedings_SS_dim_reduced = reduce_dim_with_PCA(embedings_list, explained_var)

        # Determine the optimal number of clusters based on BIC
        n_components = np.arange(1, 9)
        models = [GaussianMixture(n, covariance_type='full', random_state=42, n_init=10).fit(embedings_SS_dim_reduced)
                  for n in n_components]
        bic = [model.bic(np.array(embedings_SS_dim_reduced)) for model in models]
        n_clusters = n_components[np.argmin(bic)]

        print(f"all bic: {bic}")
        print(f"ideal number of clusters: {n_clusters}")

        # Gaussian Mixture Clustering with optimal number of clusters
        gmm = GaussianMixture(n_components=n_clusters, random_state=0, n_init=25)

        gmm.fit(embedings_SS_dim_reduced)
        return gmm.predict(np.array(embedings_SS_dim_reduced))

    # *****************
    #       CLS
    # *****************
    CLSs_cluster_labels = cluster_GMM(CLSs, explained_var)


    # *****************
    #   avg embedings
    # *****************
    avg_embs_cluster_labels = cluster_GMM(avg_embs, explained_var)


    # *****************
    #     finetune
    # *****************
    finetuned_cluster_labels = cluster_GMM(finetuned, explained_var)




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
    if summarize:
        # Define the summarizer model

        summarizer_model = Summarizer()
        # summarizer_model = SBertSummarizer('paraphrase-MiniLM-L6-v2')

        num_sentences = 3

        transition_phrases = ["Moreover, ", "Furthermore, ",
                              "In addition, ", "Similarly, ", "Also, "]
        # *****************
        #       CLS
        # *****************
        print("\n\n*** CLSs ***")
        for cluster_label, cluster_data in CLSs_clusters.items():
            print(f"\nCluster label: {cluster_label}")
            summary = ""
            for idx, (article, _) in enumerate(cluster_data):
                summarized_text = summarize_text(article['text'], summarizer_model)

                if "—" in summarized_text:
                    summarized_text = summarized_text.split("—")[1]

                summarized_text = summarized_text.replace("\n", " ")

                if idx > 0:
                    summary += transition_phrases[idx % len(transition_phrases)] + summarized_text + ". "
                else:
                    summary += summarized_text + ". "
            # print(f"Summarized text: {summary}")

            # Generate a final summary for the whole cluster
            final_summary = summarize_text(summary, summarizer_model, num_sentences=num_sentences)
            print(f"Final Summary for cluster: {final_summary}")

        # *****************
        #   avg embedings
        # *****************
        print("\n\n*** avg embedings ***")
        for cluster_label, cluster_data in avg_embs_clusters.items():
            print(f"\nCluster label: {cluster_label}")
            summary = ""
            for idx, (article, _) in enumerate(cluster_data):
                summarized_text = summarize_text(article['text'], summarizer_model)

                if "—" in summarized_text:
                    summarized_text = summarized_text.split("—")[1]

                summarized_text = summarized_text.replace("\n", " ")

                if idx > 0:
                    summary += transition_phrases[idx % len(transition_phrases)] + summarized_text + ". "
                else:
                    summary += summarized_text + ". "
            # print(f"Summarized text: {summary}")

            # Generate a final summary for the whole cluster
            final_summary = summarize_text(summary, summarizer_model, num_sentences=num_sentences)
            print(f"Final Summary for cluster: {final_summary}")

        # *****************
        #     finetune
        # *****************
        print("\n\n*** finetune ***")
        for cluster_label, cluster_data in finetuned_clusters.items():
            print(f"\nCluster label: {cluster_label}")
            summary = ""
            for idx, (article, _) in enumerate(cluster_data):
                summarized_text = summarize_text(article['text'], summarizer_model)

                if "—" in summarized_text:
                    summarized_text = summarized_text.split("—")[1]

                summarized_text = summarized_text.replace("\n", " ")

                if idx > 0:
                    summary += transition_phrases[idx % len(transition_phrases)] + summarized_text + ". "
                else:
                    summary += summarized_text + ". "
            # print(f"Summarized text: {summary}")

            # Generate a final summary for the whole cluster
            final_summary = summarize_text(summary, summarizer_model, num_sentences=num_sentences)
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
        CLSs, labels, CLSs_cluster_labels, f"CLS Embedding event {event_id} Plot", creat_video=creat_video)
    plot_event(avg_embs, labels, avg_embs_cluster_labels,
               f"AVG Embedding event {event_id} Plot", creat_video=creat_video)
    plot_event(finetuned, labels, finetuned_cluster_labels,
               f"Finetuned Embedding event {event_id} Plot", creat_video=creat_video)



def compare_multiple_events(event_idxs, data, creat_video=False):

    event_articals_list = [(data[idx]["articles"], idx) for idx in event_idxs]

    for event_idx in event_idxs:
        print(f"event index {event_idx}, artical count: {len(data[event_idx]['articles'])}")

    print("")

    for event_idx in event_idxs:
        print(f"event index {event_idx}, summary: {data[event_idx]['summary']}")

    CLSs = []
    avg_embs = []
    finetuned = []
    true_cluster_lables = []

    for (event_articals, event_idx) in event_articals_list:
        for article in event_articals:
            temp_CLS = article["CLS"]
            temp_avg_emb = article["avg_embedings"]
            temp_finetuned = article["finetuned"]

            CLSs.append(np.array(temp_CLS))
            avg_embs.append(np.array(temp_avg_emb))
            finetuned.append(np.array(temp_finetuned))
            true_cluster_lables.append(event_idx)

    # Clustering algorithm here.
    explained_var = 0.95

    def reduce_dim_with_PCA(data, explained_var):

        pca = PCA()
        data_SS = StandardScaler().fit_transform(np.array(data))
        pca.fit(data_SS)

        for idx, i in enumerate(np.cumsum(pca.explained_variance_ratio_)):
            if i > explained_var:
                reduced_dim = idx + 1
                break

        pca = PCA(n_components=reduced_dim)

        data_SS_dim_reduced = pca.fit_transform(data_SS)
        pca.fit(data_SS)

        return data_SS_dim_reduced

    def cluster_GMM(embedings_list, explained_var):
        embedings_SS_dim_reduced = reduce_dim_with_PCA(embedings_list, explained_var)

        # Determine the optimal number of clusters based on BIC
        n_components = np.arange(1, 20)
        models = [GaussianMixture(n, covariance_type='full', random_state=42, n_init=10).fit(embedings_SS_dim_reduced)
                  for n in n_components]
        bic = [model.bic(np.array(embedings_SS_dim_reduced)) for model in models]
        n_clusters = n_components[np.argmin(bic)]

        print(f"all bic: {bic}")
        print(f"ideal number of clusters: {n_clusters}")

        # Gaussian Mixture Clustering with optimal number of clusters
        gmm = GaussianMixture(n_components=n_clusters, random_state=0, n_init=25)

        gmm.fit(embedings_SS_dim_reduced)
        return gmm.predict(np.array(embedings_SS_dim_reduced))

    # *****************
    #       CLS
    # *****************
    CLSs_cluster_labels = cluster_GMM(CLSs, explained_var)
    homogeneity_score_CLS = homogeneity_score(true_cluster_lables, CLSs_cluster_labels)


    # *****************
    #   avg embedings
    # *****************
    avg_embs_cluster_labels = cluster_GMM(avg_embs, explained_var)
    homogeneity_score_avg_emb = homogeneity_score(true_cluster_lables, avg_embs_cluster_labels)


    # *****************
    #     finetune
    # *****************
    finetuned_cluster_labels = cluster_GMM(finetuned, explained_var)
    homogeneity_score_finetuned = homogeneity_score(true_cluster_lables, finetuned_cluster_labels)


    print(f"Homogenity score of CLS embedings: {homogeneity_score_CLS}")
    print(f"Homogenity score of average embedings: {homogeneity_score_avg_emb}")
    print(f"Homogenity score of finetuned embedings: {homogeneity_score_finetuned}")



def compare_multiple_events_plus(event_idxs, data, creat_video=False):

    event_articals_list = [(data[idx]["articles"], idx) for idx in event_idxs]

    for event_idx in event_idxs:
        print(f"event index {event_idx}, artical count: {len(data[event_idx]['articles'])}")

    print("")

    for event_idx in event_idxs:
        print(f"event index {event_idx}, summary: {data[event_idx]['summary']}")

    CLSs = []
    avg_embs = []
    finetuned_0 = []
    finetuned_1 = []
    finetuned_2 = []
    finetuned_3 = []
    true_cluster_lables = []

    for (event_articals, event_idx) in event_articals_list:
        for article in event_articals:
            temp_CLS = article["CLS"]
            temp_avg_emb = article["avg_embedings"]
            temp_finetuned_0 = article["finetuned_0"]
            temp_finetuned_1 = article["finetuned_1"]
            temp_finetuned_2 = article["finetuned_2"]
            temp_finetuned_3 = article["finetuned_3"]

            CLSs.append(np.array(temp_CLS))
            avg_embs.append(np.array(temp_avg_emb))
            finetuned_0.append(np.array(temp_finetuned_0))
            finetuned_1.append(np.array(temp_finetuned_1))
            finetuned_2.append(np.array(temp_finetuned_2))
            finetuned_3.append(np.array(temp_finetuned_3))

            true_cluster_lables.append(event_idx)

    # Clustering algorithm here.
    explained_var = 0.95

    def reduce_dim_with_PCA(data, explained_var):

        pca = PCA()
        data_SS = StandardScaler().fit_transform(np.array(data))
        pca.fit(data_SS)

        for idx, i in enumerate(np.cumsum(pca.explained_variance_ratio_)):
            if i > explained_var:
                reduced_dim = idx + 1
                break

        pca = PCA(n_components=reduced_dim)

        data_SS_dim_reduced = pca.fit_transform(data_SS)
        pca.fit(data_SS)

        return data_SS_dim_reduced

    def cluster_GMM(embedings_list, explained_var):
        embedings_SS_dim_reduced = reduce_dim_with_PCA(embedings_list, explained_var)

        # Determine the optimal number of clusters based on BIC
        n_components = np.arange(1, 75)
        models = [GaussianMixture(n, covariance_type='full', random_state=42, n_init=10).fit(embedings_SS_dim_reduced)
                  for n in n_components]
        bic = [model.bic(np.array(embedings_SS_dim_reduced)) for model in models]
        n_clusters = n_components[np.argmin(bic)]

        print(f"all bic: {bic}")
        print(f"ideal number of clusters: {n_clusters}")

        # Gaussian Mixture Clustering with optimal number of clusters
        gmm = GaussianMixture(n_components=n_clusters, random_state=0, n_init=25)

        gmm.fit(embedings_SS_dim_reduced)
        return gmm.predict(np.array(embedings_SS_dim_reduced))

    # *****************
    #       CLS
    # *****************
    CLSs_cluster_labels = cluster_GMM(CLSs, explained_var)
    homogeneity_score_CLS = homogeneity_score(true_cluster_lables, CLSs_cluster_labels)


    # *****************
    #   avg embedings
    # *****************
    avg_embs_cluster_labels = cluster_GMM(avg_embs, explained_var)
    homogeneity_score_avg_emb = homogeneity_score(true_cluster_lables, avg_embs_cluster_labels)


    # *****************
    #     finetune_0
    # *****************
    finetuned_cluster_labels_0 = cluster_GMM(finetuned_0, explained_var)
    homogeneity_score_finetuned_0 = homogeneity_score(true_cluster_lables, finetuned_cluster_labels_0)

    # *****************
    #     finetune_1
    # *****************
    finetuned_cluster_labels_1 = cluster_GMM(finetuned_1, explained_var)
    homogeneity_score_finetuned_1 = homogeneity_score(true_cluster_lables, finetuned_cluster_labels_1)

    # *****************
    #     finetune_2
    # *****************
    finetuned_cluster_labels_2 = cluster_GMM(finetuned_2, explained_var)
    homogeneity_score_finetuned_2 = homogeneity_score(true_cluster_lables, finetuned_cluster_labels_2)

    # *****************
    #     finetune_3
    # *****************
    finetuned_cluster_labels_3 = cluster_GMM(finetuned_3, explained_var)
    homogeneity_score_finetuned_3 = homogeneity_score(true_cluster_lables, finetuned_cluster_labels_3)



    print(f"Homogenity score of CLS embedings: {homogeneity_score_CLS}")
    print(f"Homogenity score of average embedings: {homogeneity_score_avg_emb}")
    print(f"Homogenity score of finetuned_0 embedings: {homogeneity_score_finetuned_0}")
    print(f"Homogenity score of finetuned_1 embedings: {homogeneity_score_finetuned_1}")
    print(f"Homogenity score of finetuned_2 embedings: {homogeneity_score_finetuned_2}")
    print(f"Homogenity score of finetuned_3 embedings: {homogeneity_score_finetuned_3}")


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
