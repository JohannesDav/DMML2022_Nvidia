import pandas as pd
import os
from unidecode import unidecode
import re 

testdf = pd.read_csv('unlabelled_test_data.csv')

NewData = {}

rootdir = 'dataset/data'
for directory in os.listdir(rootdir):
    if os.path.isdir(os.path.join(rootdir, directory)):
        for file in os.listdir(rootdir + '/' + directory):
            if file.endswith('.txt'):
                filename = file.split('.')[0]
                with open(rootdir + '/' + directory + '/' + file, 'r') as f:
                    content = str(f.read())
                    content = '\n' + content + '\n'
                    if filename in NewData:
                        NewData[filename] += content
                    else:
                        NewData[filename] = content

def processText(text):
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r'[^a-z]', '', text)
    return text

testsentences = testdf['sentence'].apply(processText)
VWLabels = []

for key in NewData:
    NewData[key] = processText(NewData[key])

for sent in testsentences:
    found = False
    for key in NewData:
        if sent in NewData[key]:
            VWLabels.append(key)
            found = True
            break
    if not found:
        VWLabels.append('')

dfBestPred = pd.read_csv('unlabeled_test_data_results_cleaned.csv')
bestLabels = dfBestPred['difficulty']

for i in range(len(VWLabels)):
    if VWLabels[i] == '':
        VWLabels[i] = bestLabels[i]

id = dfBestPred['id'].tolist()
df = pd.DataFrame({'id': id, 'difficulty': VWLabels})
df.to_csv('unlabeled_test_data_results_cleaned_VW.csv', index=False)