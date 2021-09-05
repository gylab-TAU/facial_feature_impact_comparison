import mlflow
import pandas as pd
import const

if __name__ == '__main__':

    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    metrics = ['train acc', 'train loss', 'val acc', 'val loss']
    # #60 epochs
    # runs = {'8b54a56beb2f4c68877b209802ae2b01': 'objects_fc6',
    #         '84bb18a4b27643dca45e2f928e1ba0a0': 'objects_conv1',
    #         '63dbf9cb202f484fa472d8aa7ca00be7': 'faces_fc6',
    #         '2f4134f1a3ed4967b74e5fe7b040960f': 'faces_conv1'}
    #20 epochs
    runs = {'cdd2f85d869f47749fdd97356c2a7f36': 'objects_fc6',
            '2de2cbbfd162419aa08b608caa9af28e': 'objects_conv1',
            'f9faa9e3ba8649eeac806c2ffd00e9b2': 'faces_fc6',
            'c2238d70444246cca88b279183614c5a': 'faces_conv1'}
    cols = []
    for run_id in runs:
        for metric in metrics:
            cols.append(f"{runs[run_id]}-{metric}")
    df = pd.DataFrame(columns=cols, index=pd.Series([i for i in range(20)]))

    print(df)

    for run_id in runs:
        for metric in metrics:
            col = f"{runs[run_id]}-{metric}"
            metric_hist = client.get_metric_history(run_id, metric)
            for measure in metric_hist:
                df.loc[measure.step][col] = measure.value
    print(df)
    df.to_csv('/home/ssd_storage/experiments/birds/training_metrics_20_reduce_10.csv')
