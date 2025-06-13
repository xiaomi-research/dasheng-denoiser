import os
import torch
from torch import nn
from vocos.feature_extractors import FeatureExtractor
from vocos.heads import FourierHead
from vocos.models import Backbone
from vocos.pretrained import instantiate_class

from dasheng_denoiser.models import EseEncoder


class denoiser(nn.Module):
    """
    The denoiser class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components:
    a feature extractor, a second_encoder, a backbone, and a head.
    load feature_extractor, backbone and head parameters from a pretrained model
    load second_encoder parameters from another pretrained model
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        second_encoder: EseEncoder,
        backbone: Backbone,
        head: FourierHead,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.second_encoder = second_encoder
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_ckpt(cls, model_path: str = None):
        """
        Class method to create a new denoiser model instance from a pre-trained model stored in the desk.
        """
        if model_path is not None and os.path.exists(model_path):
            dump = torch.load(model_path, map_location="cpu")
        else:
            dump = torch.hub.load_state_dict_from_url(
                "https://zenodo.org/records/15541088/files/dasheng-denoiser_checkpoint.pt?download=1",
                map_location="cpu",
            )

        config = dump["config"]
        feature_extractor = instantiate_class(args=(), init=config["feature_extractor"])
        second_encoder = instantiate_class(args=(), init=config["second_encoder"])
        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(feature_extractor=feature_extractor, second_encoder=second_encoder, backbone=backbone, head=head)

        state_dict = dump["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.
        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features = self.feature_extractor(audio_input)
        features = self.second_encoder(features)
        x = self.backbone(features)
        audio_output = self.head(x)
        return audio_output
