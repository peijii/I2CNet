import torch.nn as nn

class Model_E_A(nn.Module):

    def __init__(self, extractor, adjustor):
        super(Model_E_A, self).__init__()
        self.extractor = extractor
        self.adjustor = adjustor

    def forward(self, x, y):
        _, extractor_out2 = self.extractor(x)
        adjustor_out = self.adjustor(extractor_out2, y)
        return adjustor_out