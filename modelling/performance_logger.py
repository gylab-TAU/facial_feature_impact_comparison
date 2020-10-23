import pandas as pd
import const

class PerformanceLogger(object):
    def __init__(self, log_path: str):
        self.__performance_log = pd.DataFrame(columns=[const.EPOCH, const.LAYER])
        self.__performance_log.set_index([const.EPOCH, const.LAYER])
        self.__log_path = log_path

    def log_performance(self, epoch: int, performance_df: pd.DataFrame):
        for layer in performance_df.index:
            self.__performance_log[epoch, layer] = performance_df.loc[layer]
        self.__performance_log.to_csv(self.__log_path)
