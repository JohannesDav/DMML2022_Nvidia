import pandas as pd
import openai
import tqdm

df = pd.read_csv('unlabelled_test_data.csv')
sentences = df['sentence'].tolist()

ft_model = 'your model'
results = []
for s in tqdm.tqdm(sentences):
    try:
        res = openai.Completion.create(model=ft_model, prompt=s + ' ->', max_tokens=1, temperature=0)
        res = res['choices'][0]['text']
        results.append(res)
    except Exception as e:
        print(e)
        break

# create dataframe with id, sentence and difficulty
id = df['id'].tolist()
df = pd.DataFrame({'id': id, 'sentence' : sentences, 'difficulty': results})
df.to_csv('gpt3/unlabeled_test_data_results.csv', index=False)