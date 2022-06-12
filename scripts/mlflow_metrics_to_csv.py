import mlflow
import pandas as pd
import const

if __name__ == '__main__':
    epoch = 20
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    metrics = ['train acc', 'train loss', 'val acc', 'val loss']

    epochs_to_runs = {60: {'8af220a9645d4e06a5b419be24bc4a69': 'objects_fc6',
                            '3a2587a8b88f417e9eef99433f439287': 'objects_conv3',
                            'cc93ad6b911641208ee8ddf41c67003b': 'faces_fc6',
                            '1c0eb5ff60e74fd892d82c10347ed3df': 'faces_conv3'},
                      20: {'09b3184c08dd4f7a84200d84f7456876': 'objects_fc6',
                            'c34ef7709b054edd8d8cd1e6eeb85e5d': 'objects_conv3',
                            '755a0c4ab6a04e2591f28060823365de': 'faces_fc6',
                            '34c464022b0d42329f60f9e27156b15a': 'faces_conv3'}
                      }

    runs = epochs_to_runs[epoch]
    cols = []
    for run_id in runs:
        for metric in metrics:
            cols.append(f"{runs[run_id]}-{metric}")
    df = pd.DataFrame(columns=cols, index=pd.Series([i for i in range(epoch+1)]))

    print(df)

    for run_id in runs:
        for metric in metrics:
            col = f"{runs[run_id]}-{metric}"
            metric_hist = client.get_metric_history(run_id, metric)
            for measure in metric_hist:
                print(measure.step)
                print(col)
                df.loc[measure.step][col] = measure.value
    print(df)
    df.to_csv(f'/home/ssd_storage/experiments/birds/training_metrics_{epoch}_reduce_{epoch-10}.csv')
