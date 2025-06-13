from dasheng.pretrained.pretrained import Dasheng
from vocos.feature_extractors import FeatureExtractor


class Dasheng_init(Dasheng):
    @classmethod
    def form_init(cls, **additional_model_kwargs):
        """
        Class method to create a new Dasheng model instance with initialized parameters.
        """
        instance = cls(**{**additional_model_kwargs})
        return instance


def dasheng_base(**model_kwargs):
    model_kwargs["embed_dim"] = 768
    model_kwargs["depth"] = 12
    model_kwargs["num_heads"] = 12
    return Dasheng_init.form_init(**model_kwargs)


class DashengFeatures(FeatureExtractor):
    def __init__(
        self,
        dasheng_model: str = "dasheng_base",
    ):
        super().__init__()
        if dasheng_model == "dasheng_base":
            feat_encoder = dasheng_base()
        else:
            raise ValueError(f"Unsupported dasheng_model: {dasheng_model}.")
        self.feat_encoder = feat_encoder

        for param in self.feat_encoder.parameters():
            param.requires_grad = False
        self.feat_encoder.eval()

    def forward(self, audio):
        features = self.feat_encoder(audio)
        return features.transpose(1, 2)  # (B, F, T)
