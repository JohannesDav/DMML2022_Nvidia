import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('training_data.csv')
print(df.head())

sentences = df['sentence'].tolist()

labels = df['difficulty'].tolist()
le = LabelEncoder()
le.fit(["A1", "A2", "B1", "B2", "C1", "C2"])
labels = le.transform(labels)
labels = labels.tolist()


df = pd.DataFrame({'prompt': sentences, 'completion': labels})
df.to_json("gpt3/cefr.jsonl", orient='records', lines=True)


### Bash commands to prepare the dataset

# export OPENAI_API_KEY="your key"
# openai tools fine_tunes.prepare_data -f cefr.jsonl

### Bash command to train the model, first test with ada

# openai api fine_tunes.create -t "cefr_prepared_train.jsonl" -v "cefr_prepared_valid.jsonl" --compute_classification_metrics --classification_n_classes 6 -m ada

### Bash command to train the model with curie and the full dataset

# openai api fine_tunes.create -t "cefr_prepared.jsonl" -v "cefr_prepared.jsonl" --compute_classification_metrics --classification_n_classes 6 -m curie
