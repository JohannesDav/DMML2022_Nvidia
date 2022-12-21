from googleapiclient.discovery import build
import json
import pandas as pd
import os
import time
import json

my_api_key = "AIzaSyAN1qRdyX1jLUZQUdUx8B7gmaqL7x_7JpQ"
my_cse_id = "b0a582a187f164bda"


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    if "items" not in res:
        return []
    return res["items"]


sentenceFiles = ["unlabelled_test_data", "training_data"]
sentences0 = pd.read_csv(sentenceFiles[0] + ".csv")
sentences0 = sentences0["sentence"].tolist()
sentences0 = [" ".join(sentence.split()[:25]) for sentence in sentences0]
sentences0 = ['"' + sentence + '"' for sentence in sentences0]

sentences1 = pd.read_csv(sentenceFiles[1] + ".csv")
sentences1 = sentences1["sentence"].tolist()
sentences1 = [" ".join(sentence.split()[:25]) for sentence in sentences1]
sentences1 = ['"' + sentence + '"' for sentence in sentences1]
sentences = [sentences0, sentences1]


for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        outputFile = "results/" + sentenceFiles[i] + str(j) + ".txt"
        if not os.path.isfile(outputFile):
            term = sentences[i][j]
            print(sentenceFiles[i], j)
            results = google_search(term, my_api_key, my_cse_id, num=10)
            with open(outputFile, 'w') as fout:
                json.dump(results, fout)
            time.sleep(0.7) # comply with queries per minute quota