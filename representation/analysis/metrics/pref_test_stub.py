
class PerformanceTestStub(object):
    def calc_performance(self, y_scores, y_true):
        return y_scores

    def get_metric_name(self):
        return ["raw scores"]
