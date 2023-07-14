# csnlp-project

This project was made for the ETH Zürich Computational Semantics for Natural Language Processing course.


* Summary of the project:
The news industry has been revolutionized with the digitization of media, and the Internet becoming a widely used tool. Many news articles are generated at any moment and cause a data mess. In our project, we implemented a tool that automatically clusters news articles according to the event they relate to and their perspective on that specific event. We used BART, which is finetuned for news summarization, to generate embedding vectors for articles. We then finetuned the BART model with an auxiliary binary classification task, where our model learns whether a doublet of articles covers the same event. Then, we fed these finetuned embeddings into a Gaussian Mixture Model to cluster the articles. Ultimately, we implemented a multi-text summarization method to utilize a more generic summary of the clusters. We used clustering with BART embeddings as our baseline and compared our finetuned embeddings with them. The clustering results of finetuned embeddings yielded a much better clustering performance at clustering events based on the article they relate to compared to directly using BART embeddings.

* Under Plot_animations you can find example scatter plot 3D animations for different embedding types.
* Under Additional_Results you can find extra results that are not shown in the report due to page limit.

* To reproduce the results:
1) Download the repository.
2) download the Large-Scale Multi-Document Summarization Dataset from the Wikipedia Current Events Portal from "https://github.com/complementizer/wcep-mds-dataset" and save the 3 zip files into a folder named "Data".
3) you can directly run the main function to get the results.
4) To re-train / train new encoders, uncomment the training functions in the main function.

* What each script is for:
1) Main: is the mains script that calls all the other scripts.
2) Preprocessing: responsible for extracting data and preprocessing the data then saving it as a JSON file
3) BART_embeding_generation: responsible for creating CLS token embedding and average token embedding for all articles and appending them to their respective articles in the JSON, then saving the new data.
4) finetunning_data_generator: responsible for finetuning the BART model for article clustering than saving the encoder that does the finetuning. The encoder will be later used by other scripts for generating finetuned embeddings.
5) BART_finetuned_embeding_generation: responsible for using the saved encoders for generating finetuned embeddings for data with CLS token embeddings then appending the new embeddings to the data and saving it.
6) result_ploting: all the functions used to analyze the data and create the plots and summaries. 
7) untitled0.ipynb: the jupyter notebook where auxiliary training is performed
