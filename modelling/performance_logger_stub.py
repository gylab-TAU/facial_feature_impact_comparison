

class PerformanceLoggerStub(object):
    """empty performance logger if we don't need to save model's performance"""
    def log_performance(self, epoch, perf):
        return
