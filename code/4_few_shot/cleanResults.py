import pandas as pd 

df = pd.read_csv('gpt3/few-shot/unlabeled_test_data_results.csv')

difficulties = df['difficulty'].tolist()
difficulties = [5 if d == 6 else d for d in difficulties]
levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
difficulties = [levels[d] for d in difficulties]

id = df['id'].tolist()

df = pd.DataFrame({'id': id, 'difficulty': difficulties})
df.to_csv('gpt3/few-shot/unlabeled_test_data_results_cleaned.csv', index=False)