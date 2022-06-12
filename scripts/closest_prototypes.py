import pandas as pd
import plotly
import numpy as np

if __name__ == '__main__':
    path = '/home/administrator/experiments/familiarity/pretraining/vgg16/results/fc8_dist_mat.csv'
    df = pd.read_csv(path)
    df = df.set_index(['Unnamed: 0'])

    stacked_df = df.stack()
    stacked_df = stacked_df.reset_index()
    stacked_df = stacked_df.rename(columns={"Unnamed: 0": 'cls1', "level_1": "cls2", 0: "cosine sim"})
    stacked_df.set_index(['cls1', 'cls2'])

    print('similar')
    print(stacked_df.sort_values('cosine sim', ascending=False)[8749:8759])
    print('dissimilar')
    print(stacked_df.sort_values('cosine sim', ascending=True)[:10])

