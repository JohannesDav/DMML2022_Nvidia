import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json


df = pd.read_csv('training_data.csv')
print(df.head())

sentences = df['sentence'].tolist()

noapicount = 0
noresultsCount = 0
googleResults = []
for i in range(len(sentences)):
    filename = "googleScrape/results/training_data" + str(i) + ".txt"
    with open(filename) as fin:
        results = json.load(fin)
    if len(results) > 0:
        textRes = ''
        for result in results:
            res = result['title'] + ' --- ' + result['formattedUrl']
            res = res.rstrip()
            textRes += res + '\n'
        googleResults.append(textRes)
    else:
        noapicount += 1
        filename2 = "googleScrape/results/training_data" + str(i) + "_scraped.txt"
        with open(filename2, 'r') as file:
            data = file.read()
        if len(data) > 0:
            googleResults.append(data)
        else:
            googleResults.append('No results\n')
            noresultsCount += 1

# for g in googleResults:
#     print(g)
# print("% missing without scrape:", noapicount/len(sentences))
# print("% missing with scrape:", noresultsCount/len(sentences))


promts = []
for i in range(len(sentences)):
    prompt = "Sentence: " + sentences[i] + "\nSearch results:\n" + googleResults[i] + "Level:"
    promts.append(prompt)


labels = df['difficulty'].tolist()
le = LabelEncoder()
le.fit(["A1", "A2", "B1", "B2", "C1", "C2"])
labels = le.transform(labels)
labels = labels.tolist()


df = pd.DataFrame({'prompt': promts, 'completion': labels})
df.to_json("gpt3/cefr_google.jsonl", orient='records', lines=True)
df.to_csv("gpt3/cefr_google.csv", index=False)


### Bash commands to prepare the dataset

# export OPENAI_API_KEY="your key"
# openai tools fine_tunes.prepare_data -f cefr_google.jsonl

### Bash command to train the model

# openai api fine_tunes.create -t "cefr_google_prepared.jsonl" -v "cefr_google_prepared.jsonl" --compute_classification_metrics --classification_n_classes 6 -m curie