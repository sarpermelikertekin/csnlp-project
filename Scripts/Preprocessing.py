import json
import gzip




def read_jsonl_gz(path):
    with gzip.open(path) as f:
        for l in f:
            yield json.loads(l)

def generate_data():
    val_data = list(read_jsonl_gz('val.jsonl.gz'))
    test_data = list(read_jsonl_gz('test.jsonl.gz'))
    train_data = list(read_jsonl_gz('train.jsonl.gz'))

    data = train_data + test_data + val_data


    list_of_good_sites = ["www.dailymail.co.uk", "www.reuters.com", "www.channelnewsasia.com", "www.cnn.com",
                          "in.reuters.com", "www.seattletimes.com", "www.express.co.uk", "www.foxnews.com",
                          "www.theguardian.com", "nationalpost.com", "abcnews.go.com", "uk.reuters.com",
                          "sputniknews.com", "www.westernjournal.com", "www.nbcnews.com", "www.npr.org",
                          "www.washingtonpost.com", "indianexpress.com", "www.bbc.com", "www.metronews.ca",
                          "time.com", "www.businessinsider.com", "www.irishtimes.com", "www.firstpost.com",
                          "www.firstpost.com", "www.straitstimes.com", "www.aljazeera.com", "www.voanews.com",
                          "www.thenews.com.pk", "www.startribune.com", "www.yahoo.com", "www.ctvnews.ca",
                          "www.myplainview.com", "www.digitaljournal.com", "www.sfgate.com", "www.theglobeandmail.com",
                          "nation.com.pk", "www.dailysabah.com", "www.nytimes.com", "www.bbc.co.uk",
                          "www.newindianexpress.com", "www.nytimes.com", "www.washingtontimes.com", "www.cnbc.com"]

    list_of_good_categories = ["Politics and elections", "Law and crime", "Disasters and accidents",
                              "Armed conflicts and attacks", "International relations", "Business and economy",
                              "Science and technology"]

    # removeing not active sites.
    new_data = []
    for event in data:
        category = event["category"]

        if category in list_of_good_categories:

            articles_list = event["articles"]
            new_articles_list = []
            for article in articles_list:
                url = article["url"]

                if url and ("//" in url):
                    website = (url.split("//")[1]).split("/")[0]
                else:
                    website = "None"

                if website in list_of_good_sites:

                    new_articles_list.append(article)

            event["articles"] = new_articles_list

            new_data.append(event)
    data = new_data



    # removeing not well covered events.
    new_data = []
    for event in data:
        articles_list = event["articles"]
        if len(articles_list) > 10:

            new_data.append(event)
    data = new_data



    news_site_dict = {}
    category_dict = {}
    coverage_dict = {}
    for event in data:
        category = event["category"]

        if (category != "Sports") and (category != "Arts and culture"):
            if category in category_dict:
                category_dict[category] += 1
            else:
                category_dict[category] = 1

            articles_list = event["articles"]

            coverage = len(articles_list)
            if coverage in coverage_dict:
                coverage_dict[coverage] += 1
            else:
                coverage_dict[coverage] = 1

            for article in articles_list:
                url = article["url"]

                if url and ("//" in url):
                    website = (url.split("//")[1]).split("/")[0]
                else:
                    website = "None"

                if website in news_site_dict:
                    news_site_dict[website] += 1
                else:
                    news_site_dict[website] = 1

    news_site_list = sorted(news_site_dict.items(), key=lambda x: x[1], reverse=True)
    category_list = sorted(category_dict.items(), key=lambda x: x[1], reverse=True)
    coverage_list = sorted(coverage_dict.items(), key=lambda x: x[0], reverse=True)


    for i in news_site_list:
        if i[1] > 1000:
            print(f"{i[0]}: {i[1]:5d}")

    print("\n\nCategoreis:")
    for i in category_list:
        if i[1] > 100:
            print(f"{i[0]}: {i[1]:5d}")

    print("\n\nCoverage rates:")
    for i in coverage_list:
        if i[1] > 100:
            print(f"{i[0]}: {i[1]:5d}")

    jsonString = json.dumps(data)
    jsonFile = open("data.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print("debug")


