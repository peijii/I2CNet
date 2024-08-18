import torch.nn as nn

class Model_E_P(nn.Module):

    def __init__(self, extractor, predictor):
        super(Model_E_P, self).__init__()
        self.extractor = extractor
        self.predictor = predictor

    def forward(self, x):
        extractor_out1, extractor_out2 = self.extractor(x)
        predictor_out = self.predictor(extractor_out1)
        return predictor_out, extractor_out2