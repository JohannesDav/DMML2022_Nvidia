import pandas as pd
from sklearn.preprocessing import LabelEncoder
import openai
import tqdm
import json

df = pd.read_csv('unlabelled_test_data.csv')
sentences = df['sentence'].tolist()

googleResults = []
for i in range(len(sentences)):
    filename = "googleScrape/results/unlabelled_test_data" + str(i) + ".txt"
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
        filename2 = "googleScrape/results/unlabelled_test_data" + str(i) + "_scraped.txt"
        with open(filename2, 'r') as file:
            data = file.read()
        if len(data) > 0:
            googleResults.append(data)
        else:
            googleResults.append('No results\n')

promts = []
for i in range(len(sentences)):
    prompt = "Sentence: " + sentences[i] + "\nSearch results:\n" + googleResults[i] + "Level:"
    promts.append(prompt)


ft_model = 'your model'
results = []
for s in tqdm.tqdm(promts):
    try:
        res = openai.Completion.create(model=ft_model, prompt=s, max_tokens=1, temperature=0)
        res = res['choices'][0]['text']
        results.append(res)
    except Exception as e:
        print(e)
        results.append('')

# create dataframe with id, sentence and difficulty
id = df['id'].tolist()
df = pd.DataFrame({'id': id, 'sentence' : promts, 'difficulty': results})
df.to_csv('gpt3/unlabeled_test_data_results.csv', index=False)
