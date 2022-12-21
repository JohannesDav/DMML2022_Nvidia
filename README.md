
# DMML22: Detecting the difficulty level of French texts Project


## Participants

Team NVIDIA: Johannes Rudolf David & Emmanuel Hubert

## Concept
The task of this project was to identify the difficulty of French text, according to the six CEFR levels.
In the first file (Project_NVIDIA) we applied standard machine learning methods and obtained our first scores.
In the second part we tried finding a more creative approach to obtain better accuracy in our results.

## Approach description

### 1. CNN and LSTM trained from scratch

### 2. Camembert fine-tuning

### 3. GPT-3 fine-tuning

### 4. GPT-3-davinci few-shot

### 5. GPT-3 fine-tuning with Google search results


First, we used model XXX

Then we applied the following data cleaning XXX

In addition to that, we used embeddings XXX

In addition to data cleaning XXX, we used the Google API to backtrack the source of the text in the database. Through this method, the model learns the general difficulty level of the source, giving the text originating from said source an 80% chance to be identical.

## Summary of the results

|           | Logistic regression |      kNN      | Decision Tree | Random Forests | Our method |
|-----------|---------------------|---------------|---------------|----------------|------------|
| Accuracy  |0.40417|0.31875|0.30833|0.42708|0.74333|
| Precision |0.41691|0.40304|0.30970|0.43851	|-|
|   Recall  |0.40417|0.31875|0.30833|0.42708	|-|
|  F1-score |0.39165|0.30217|0.30529|0.41404|-|

Thanks to this table, it is clearly visible that our model produces far better results than any of the simpler methods. From the first four, the Logistic Regression produces the best results.

## Video

http

## License

None
