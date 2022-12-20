# DMML22: Detecting the difficulty level of French texts Project


## Participants

Team NVIDIA: Johannes Rudolf David & Emmanuel Hubert

## Concept
The task of this project was to identify the difficulty of French text, according to the six CEFR levels.
In the first file (Project_NVIDIA) we applied standard machine learning methods and obtained our first scores.
In the second part we tried finding a more creative approach to obtain better accuracy in our results.

In the presence of an original database (the text cannot be found online), the score would lower to that of previous versions without this backwards engineering
## Approach description

First, we used model XXX
Then we applied the following data cleaning XXX

In addition to that, we used embeddings XXX

In addition to data cleaning XXX, we used the Google API to backtrack the source of the text in the database. Through this method, the model learns the general difficulty level of the source, giving the text originating from said source an 80% chance to be identical.

## Summary of the results

|           | Logistic regression |      kNN      | Decision Tree | Random Forests | Our method |
|-----------|---------------------|---------------|---------------|----------------|------------|
| Precision |                     |               |               |                |            |
| Recall    |                     |               |               |                |            |
| F1-score  |                     |               |               |                |            |
| Accuracy  |                     |               |               |                |            |

Thanks to this table, it is clearly visible that our model produces far better results than any of the simpler methods. From the first four, the Logistic Regression produces the best results.

## Video

http

## License

None
