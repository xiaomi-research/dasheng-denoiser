import torch
from torch import nn

from dasheng.train.models import Block


class EseEncoder(nn.Module):
    """Base class for the EseEncoder. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, H, L), where B is the batch size,H denotes the model dimension
                        and L is the sequence length.

        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class DashengBlocks(EseEncoder):
    """
    Args:
        embedding_dims (int): Hidden dimension of the model.
        num_heads (int): Number of heads in Attention layers.
        num_layers (int): Number of Attention layers in Block.
    """

    def __init__(
        self,
        embedding_dims: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embedding_dims,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    qkv_bias=True,
                    init_values=None,
                    drop=0.0,
                    attn_drop=0.0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    attention_type="Attention",
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dims)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: (B, D, T)
        return: (B, D, T)

        """
        x = self.blocks(x.transpose(1, 2))
        x = self.norm(x)
        return x.transpose(1, 2)
