import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random


df = pd.read_csv('training_data.csv')
sentences = df['sentence'].tolist()
labels = df['difficulty'].tolist()
le = LabelEncoder()
le.fit(["A1", "A2", "B1", "B2", "C1", "C2"])
labels = le.transform(labels)
labels = labels.tolist()

sourceFiles =  ["dataset/CEFR-SP/SCoRE/CEFR-SP_SCoRE_dev.txt", "dataset/CEFR-SP/SCoRE/CEFR-SP_SCoRE_test.txt", "dataset/CEFR-SP/SCoRE/CEFR-SP_SCoRE_train.txt",
                "dataset/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_dev.txt", "dataset/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_test.txt", "dataset/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_train.txt"]

dfnew = pd.DataFrame()
for file in sourceFiles:
    dfnew = dfnew.append(pd.read_csv(file, sep='\t', header=None, names=['text', 'label1', 'label2']), ignore_index=True)

newSentences = dfnew['text'].tolist()
newLabels1 = dfnew['label1'].tolist()
newLabels2 = dfnew['label2'].tolist()

allSentences = []
allLabels = []
for i in range(len(newSentences)):
    sentence = newSentences[i]
    label1 = newLabels1[i]
    label2 = newLabels2[i]
    if label1 == label2: # The two annotators agreed on the label
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.replace(" .", ".")
        allSentences.append(sentence)
        allLabels.append(int(label1) - 1) # 1-6 -> 0-5 


allSentences += sentences
allLabels += labels

print(allSentences)
print(allLabels)

# suffle the two lists in the same order
c = list(zip(allSentences, allLabels))
random.shuffle(c)
allSentences, allLabels = zip(*c)

df = pd.DataFrame({'prompt': allSentences, 'completion': allLabels})
df.to_json("gpt3/cefr_bilingual.jsonl", orient='records', lines=True)

### Bash commands to prepare the dataset

# export OPENAI_API_KEY="your key"
# openai tools fine_tunes.prepare_data -f cefr_bilingual.jsonl

### Bash command to train the model

# openai api fine_tunes.create -t "cefr_bilingual_prepared.jsonl" -v "cefr_bilingual_prepared.jsonl" --compute_classification_metrics --classification_n_classes 6 -m curie
