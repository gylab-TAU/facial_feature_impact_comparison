import mlflow
import pandas as pd
import const
import plotly.express as px

def get_experiment_val_acc(experiment_name: str):
    exp_id = dict(mlflow.get_experiment_by_name(experiment_name))['experiment_id']
    runs = mlflow.search_runs(exp_id)
    name2val_acc = runs[['start_time', 'tags.mlflow.runName', 'metrics.val acc']]
    name2val_acc = name2val_acc.groupby('tags.mlflow.runName').min('start_time')
    return name2val_acc


def plot_acc(experiment_names):
    df = None
    for name in experiment_names:
        val_acc = get_experiment_val_acc(experiment_names[name])
        val_acc['pretraining'] = name
        if df is None:
            df = val_acc
        else:
            df = pd.concat([df, val_acc])
    df['depth'] = [s[0] for s in df.index.str.split('_')]
    df = df.rename(columns={"metrics.val acc": 'validation accuracy'})
    fig = px.line(df, x='depth', y='validation accuracy', color='pretraining')
    fig.update_yaxes(range=[0.0, 1.0])
    fig.write_html('/home/ssd_storage/experiments/birds/ind_birds_layer_finetune/acc@1_over_depth.html')


if __name__ == '__main__':
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    plot_acc({'Objects': 'individual_birds_objects_finetuning', 'Faces': 'individual_birds_faces_finetuning'})
