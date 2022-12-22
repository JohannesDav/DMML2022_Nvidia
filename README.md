
# DMML22: Detecting the difficulty level of French texts Project


## Participants

Team NVIDIA: Johannes-Rudolf David & Emmanuel Hubert

## Concept
The task of this project was to identify the difficulty of French text, according to the six CEFR levels.  
In the first file ([Project_NVIDIA.ipynb](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/Project_NVIDIA.ipynb "Project_NVIDIA.ipynb")) we applied standard machine learning methods and obtained our first scores.  
In the second part we tried finding a more creative approach to obtain better accuracy in our results for the [Kaggle competition](https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2022/leaderboard).  

## Approach description

### 1. CNN and LSTM trained from scratch
Sentence encoding / feature extraction: each word is represented as a vector containing either its [fastText embedding](https://fasttext.cc/docs/en/crawl-vectors.html),  its frequency in texts for each CEFR level ([FLELex](http://cental.uclouvain.be/flelex/)), or both.  
Even with hyperparameter tuning, these more complex and larger models performed only slightly better than simpler models. This indicates that using pretrained models or augmenting the dataset could be more effective than focussing only on model architecture.
Using the FLELex word frequencies provided no benefit compared to the embeddings.  

#### Accuracy
CNN word frequencies only: ~0.4  
CNN embeddings only: ~0.45  
CNN word frequencies + embeddings: ~0.43  
LSTM word frequencies + embeddings: ~0.47  

#### Files
 - [encodeData.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/1_CNN_LSTM/encodeData.py "encodeData.py") - Vectorize the sentences and save the dataset
 - [trainCNN.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/1_CNN_LSTM/trainCNN.py "trainCNN.py") - File used to train and experiment with various CNN-based models
 - [trainLSTM.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/1_CNN_LSTM/trainLSTM.py "trainLSTM.py") - File used to train and experiment with various LSTM-based models

### 2. CamemBERT fine-tuning
The pretrained CamemBERT models, available on Hugging Face, all provided a significant increase in accuracy compared the previous methods.
#### Accuracy
5 epochs fine-tuning: 0.579  
11 epochs fine-tuning: 0.586  
#### Files
 - [camemBERT.ipynb](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/2_camemBERT/camemBERT.ipynb "camemBERT.ipynb") - A notebook based on [this example](https://www.kaggle.com/code/houssemayed/camembert-for-french-tweets-classification/notebook) to fine-tune [camemBERT-large](https://huggingface.co/camembert/camembert-large) and to use the fine-tuned model to make predictions.

### 3. GPT-3 fine-tuning
The pretrained GPT-3 models, available through the OpenAI api, achieved similar accuracies as CamemBERT.  
These models are multilingual, allowing for more flexibility when experimenting with our dataset. Since using the entire training data was beneficial with camemBERT and GPT-3 Curie, we tried extending the dataset using only high quality examples from [CEFR-SP](https://github.com/yukiar/CEFR-SP/tree/main/CEFR-SP). We also scraped full texts from multiple websites offering French reading exercises labelled by CEFR level and sampled sentences from these texts. Any addition to the original competition dataset yielded worse results. Our augmented models, while perhaps more generalizable, led to lowers scores on Kaggle. This might suggest that there is some bias in the dataset.  

#### Accuracy
Ada with 80% of the dataset: 0.577  
Curie with 100% of the dataset: **0.615**  
Curie with 100% of the dataset + CEFR-SP: 0.570  

#### Files
 - [createDataset.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/3_GPT-3/createDataset.py "createDataset.py") - Used to create the training dataset for gpt-3 using the provided training sentences
 - [createDataset_bilingual.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/3_GPT-3/createDataset_bilingual.py "createDataset_bilingual.py") - Used to create the training dataset for gpt-3 using the provided training sentences and the CEFR-SP dataset

Once the CSV file with pairs of prompts and completions is created, it can be prepared for training with the following command:

    openai tools fine_tunes.prepare_data -f cefr.jsonl

The model can then be trained on the prepared dataset

    export OPENAI_API_KEY="your_key"
    openai api fine_tunes.create -t "cefr_prepared.jsonl" -v "cefr_prepared.jsonl" --compute_classification_metrics --classification_n_classes 6 -m curie

 - [classify.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/3_GPT-3/classify.py "classify.py") - To use the fine-tuned model to classify the test dataset
 - [cleanResults.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/3_GPT-3/cleanResults.py "cleanResults.py") - To format the results before submitting to Kaggle



### 4. GPT-3-davinci few-shot
By prompting GPT-3-davinci only with the first example for each class in the dataset, the accuracy achieved was already comparable to some previous methods.  
This method did not contribute to finding our winning solution, but the impressive performance of such an easy and quick method is worth mentioning. The accuracy could probably be improved considerably by adding a few more examples, by hand-picking examples that are very representative of their class or by adapting the examples to the approximated level of the sentence.  

#### Accuracy
text-davinci-003 with 6 examples: 0.402  

#### Files
 - [classify.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/4_few_shot/classify.py "classify.py") - To classify the unlabelled sentences using few-shot learning
 - [cleanResults.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/4_few_shot/cleanResults.py "cleanResults.py") - To format the results before submitting to Kaggle

### 5. GPT-3 fine-tuning with Google search results

We observed that some easy sentences were labelled as C1 or C2. They all had in common that they were extracted from sources intended for advanced speakers (classical litterature, wikipedia, C2 exams, complex news sources, advances exercices...). We also stumbled across relativelly diffcult sentences labelled as A1 or A2. These were all taken from overall simpler texts (reading exercises, A1 exams, simple news...).  
Here are a few examples :

 - C2 "Tu mangeas les petits fruits dès que tu les eus cueillis" ([complex exercise](https://oraprdnt.uqtr.uquebec.ca/pls/public/docs/GSC2213/F1880081139_Le_pass__ant_rieur_Exercices_et_corrig_.pdf))
 - C2 "Les valorisations boursières des sociétés Internet comme AMAZON" ([Thesis](https://theses.hal.science/tel-00011019/file/TheseADudezert.PDF))
 - C2 "On peut faire une classe inversée." ([DALF C2](https://www.delfdalf.ch/fileadmin/user_upload/Unterlagen/Exemples_examens/C2/2018_uploads/C2_LSH/c2-nouveau_example1_LSH_transcriptions.pdf))
 - A2 "Et en fait, le cheveu est capable d'absorber jusqu'à huit fois son poids en hydrocarbure" ([francaisfacile.rfi.fr](https://savoirs.rfi.fr/ru/apprendre-enseigner/environnement/ile-maurice-des-cheveux-contre-la-maree-noire/2))

Even for more accuratelly labeled sentences, knowing their source can be a powerful predictor of their class. This advantage is possibly amplified by the fact that sentences sampled from the same text will often be present in both the train and test datasets.  

In other words, we can say that, for the Kaggle competition, the real task is not to classify arbitratrary sentences, but only parts of existing texts for reading comprehension. Having more information about the source material makes this tasks easier.   
To approximate this information about the source text, we included the Google search results for the sentence in the prompts.  

#### Accuracy
Curie with api results: 0.718  
Curie with api results + scraped results: **0.735**  

#### Files
 - [customsearchAPI.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/5_GTP-3_Google/DataCollection/customsearchAPI.py "customsearchAPI.py") - Save search results using a programmable search engine and the Google cloud API. This method [is limited](https://support.google.com/programmable-search/answer/70392?hl=en#:~:text=Programmable%20Search%20Engines%20configured%20to%20search%20the%20entire%20web%20are%20limited%20to%20a%20subset%20of%20the%20total%20Google%20Web%20Search%20corpus.) and only found results for 72% of the dataset.
 - [scrape.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/5_GTP-3_Google/DataCollection/scrape.py "scrape.py")  - Save search results by scraping Google with selenium only for the missing results. This method allowed the dataset coverage to be increased to 83%. Sentences with no results are labeled as such, which is in itself a potentially usefull information.
 - [createDatasetGoogle.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/5_GTP-3_Google/Train/createDatasetGoogle.py "createDatasetGoogle.py") - To prepare the new prompts for training with the OpenAI api
 - [classify_google.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/5_GTP-3_Google/Train/classify_google.py "classify_google.py") - To use the fine-tuned model to classify the test dataset
 - [cleanResults.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/5_GTP-3_Google/Train/cleanResults.py "cleanResults.py") - To format the results before submitting to Kaggle

### VW module

We saw that some of the texts we had downloaded when trying to extend the dataset contained some of the sentences in our dataset. These texts are labeled with their difficulty. We observed that if a sentence is found in a text, it has an ~80% chance of being of the same diffuculty level as the text itself. About 15% of the unlabeled dataset is covered, and applying this simple search based classification on top off another model's results can improve the accuracy, especially for weaker models. This obviously exploits the fact that the model is used in a specific test enviromnent (any naming resemblance to a real company is purely coincidental).   

#### Accuracy boost
On the gtp3-curie results: 0.615 → 0.657  
On the gtp3-curie + Google api results: 0.718 → 0.727  
On the gtp3-curie + Google api + Google scrape results: 0.735 → **0.743**  

#### Files

 - [VW.py](https://github.com/JohannesDav/DMML2022_Nvidia/blob/main/code/VW/VW.py
   "VW.py") - Apply the VW method to the cleaned results


## Summary of the results

|           | Logistic regression |      kNN      | Decision Tree | Random Forests | Our method |
|-----------|---------------------|---------------|---------------|----------------|------------|
| Accuracy  |0.46667|0.31875|0.31667|0.41458|0.74333|
| Precision |0.46556|0.40304|0.31757|0.42082|-|
|   Recall  |0.46667|0.31875|0.31667|0.41458|-|
|  F1-score |0.46400|0.30217|0.31348|0.39999|-|

Thanks to this table, it is clearly visible that our model produces far better results than any of the simpler methods. From the first four, the Logistic Regression produces the best results.

## Video

http

## License

None
