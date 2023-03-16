from typing import List

import mlflow
import pandas as pd
import const
import plotly.express as px

def get_layer_auc(experiment_names: List[str], layer: str, metric: str = 'depth fc7 verification auc') -> pd.DataFrame:
    metrics_df = {}
    for exp in experiment_names:
        metrics_df[exp] = []
        exp_id = dict(mlflow.get_experiment_by_name(exp))['experiment_id']
        runs = mlflow.search_runs(exp_id)
        id2names = runs[['run_id', 'tags.mlflow.runName']]
        id2names = id2names[id2names['tags.mlflow.runName'] == f'{layer}_auc_every_epoch']
        client = mlflow.tracking.MlflowClient()
        run_id = id2names.iloc[0]['run_id']
        print(run_id)
        metric_hist = client.get_metric_history(run_id, metric)
        for m in metric_hist:
            metrics_df[exp].append(m.value)
    return pd.DataFrame(metrics_df, index=[i for i in range(1, 61)])


def plot_aucs(aucs: pd.DataFrame, layer: str):
    fig = px.line(aucs, title=layer)
    fig.update_yaxes(range=[0.0, 1.0])
    fig.show()
    fig.write_html(f'/home/ssd_storage/experiments/birds/aucs/aucs_over_epochs_{layer}.html')


if __name__ == '__main__':
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    experiments = [
        'individual_birds_faces_finetuning',
        'individual_birds_objects_finetuning',
        'birds_faces_finetuning',
        'birds_objects_finetuning'
    ]
    for layer in ['conv5', 'fc7']:
        layer_df = get_layer_auc(experiments, layer)
        layer_df.to_csv(f'/home/ssd_storage/experiments/birds/aucs/{layer}.csv')
        plot_aucs(layer_df, layer)

