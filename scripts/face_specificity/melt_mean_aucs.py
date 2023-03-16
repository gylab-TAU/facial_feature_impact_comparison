import pandas as pd


if __name__ == '__main__':
    for ds in ['sociable weavers', 'bird species']:
        for backbone in ['objects', 'faces']:
            input_path = f'/home/ssd_storage/experiments/sociable_weavers/vgg16/results/{ds}_random_sampled_{backbone}_aucs.csv'
            output_path = f'/home/ssd_storage/experiments/sociable_weavers/vgg16/results/melted/{ds}_random_sampled_{backbone}_aucs.csv'
            curr_df = pd.read_csv(input_path, index_col='Unnamed: 0')

            flattened = curr_df.stack()

            flattened.to_csv(output_path)
