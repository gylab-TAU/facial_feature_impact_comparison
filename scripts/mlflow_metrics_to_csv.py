import mlflow
import pandas as pd
import const

if __name__ == '__main__':
    epoch = 60
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    metrics = ['train acc', 'train loss', 'val acc', 'val loss']

    epochs_to_runs = {60: {'db3c6de633d440ea8986120cd40224b8': 'objects_fc6',
                            '076a9e88841b4b35acd2649b23b241d1': 'objects_conv3',
                            'bb2577c175f04300a5aef13f7225a64e': 'faces_fc6',
                            '68c8aea457be46f6b3f6dd0a6079e8dd': 'faces_conv3'},
                      20: {'e77bd511e0434882b1bbcc79b1bb0b38': 'objects_fc6',
                            '72ba103eeb7447c680eba93ed8d50003': 'objects_conv3',
                            'ef711c6a59224a17952298f1885e60d2': 'faces_fc6',
                            '40edabe593d04e5192c00435b47ca660': 'faces_conv3'}
                      }

    runs = epochs_to_runs[epoch]
    cols = []
    for run_id in runs:
        for metric in metrics:
            cols.append(f"{runs[run_id]}-{metric}")
    df = pd.DataFrame(columns=cols, index=pd.Series([i for i in range(epoch)]))

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
    df.to_csv(f'/home/ssd_storage/experiments/individual_birds/training_metrics_{epoch}_reduce_{epoch-10}.csv')
