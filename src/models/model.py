from featureExtractor import FeatureExtractor
from labelAdjustor import LabelAdjustor
from labelPredictor import LabelPredictor
from model_e_a import Model_E_A
from model_e_p import Model_E_P


class I2CNet():

    def __init__(
            self,
            in_planes: int = 10,
            num_classes: int = 5,
            mse_b1: int = 5,
            mse_b2: int = 11,
            mse_b3: int = 21,
            expansion_rate: int = 2,
            reduction_rate: int = 4,
            cell1_num: int = 1,
            cell2_num: int = 1,
    ) -> None:
        self.feature_extractor = FeatureExtractor(in_planes=in_planes, num_classes=num_classes, mse_b1=mse_b1, mse_b2=mse_b2, mse_b3=mse_b3,
                                                  expansion_rate=expansion_rate, reduction_rate=reduction_rate, cell1_num=cell1_num, cell2_num=cell2_num)
        self.label_predictor = LabelPredictor(num_classes=num_classes)
        self.label_adjustor = LabelAdjustor(num_classes=num_classes)
        self.model_e_p = self.build_model_e_p()
        self.model_e_a = self.build_model_e_a()

    def build_model_e_p(self):
        model_e_p = Model_E_P(extractor=self.feature_extractor, predictor=self.label_predictor)
        return model_e_p

    def build_model_e_a(self):
        model_e_a = Model_E_A(extractor=self.feature_extractor, adjustor=self.label_adjustor)
        return model_e_a

