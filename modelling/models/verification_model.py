from torch import nn


class VerificationModel(nn.Module):
    def __init__(self):
        self.verify = False

    def set_classify(self):
        self.verify = False

    def set_verify(self, verify=True):
        self.verify = verify
