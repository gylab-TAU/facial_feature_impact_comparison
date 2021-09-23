import mlflow
import pandas as pd
import const

if __name__ == '__main__':
    epoch = 60
    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    metrics = ['train acc', 'train loss', 'val acc', 'val loss']
    #60 epochs
    runs = {'2adab52f2c6349fc8556fb572446e44e': 'objects_fc6',
            'b2b89e5714db424f912c375e281ef1d6': 'objects_conv3',
            '4ed940a82a304fa096ead388e41bb1ce': 'faces_fc6',
            '0907a90f93cf47199105b8ea19e69a3d': 'faces_conv3'}
    # 20 epochs
    # runs = {'7088abd7a7a544a68dc1243886c12c8f': 'objects_fc6',
    #         'b6542559ab684dd3b71daa21fce253e6': 'objects_conv3',
    #         '04563b9bf28246e282287019860d1d92': 'faces_fc6',
    #         '80f8a01a45594acdb2f05987aab07a13': 'faces_conv3'}
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
                df.loc[measure.step][col] = measure.value
    print(df)
    df.to_csv(f'/home/ssd_storage/experiments/birds/training_metrics_{epoch}_reduce_{epoch-10}.csv')
