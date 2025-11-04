"""
Sparse Autoencoder (SAE) implementation.

Supports two types:
- vanilla: Uses L1 regularization for sparsity
- topk: Uses top-k activation sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder class.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    hidden_dim : int
        Dimension of the bottleneck/hidden layer.
    expanded_ratio : float
        Ratio to expand the hidden dimension. The expanded dimension will be
        hidden_dim * expanded_ratio.
    type : str, default="vanilla"
        Type of sparsity mechanism. Options: "vanilla" or "topk".
    n_encoder_layers : int, default=1
        Number of layers in the encoder.
    n_decoder_layers : int, default=1
        Number of layers in the decoder.

    Attributes
    ----------
    encoder : nn.Module
        Encoder network (multi-layer).
    expansion : nn.Linear
        Expansion layer from bottleneck to expanded dimension.
    decoder : nn.Module
        Decoder network (multi-layer).
    expanded_dim : int
        The expanded dimension (hidden_dim * expanded_ratio).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        expanded_ratio: float,
        type: str = "vanilla",
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
    ):
        super().__init__()

        if type not in ["vanilla", "topk"]:
            raise ValueError(f"type must be 'vanilla' or 'topk', got '{type}'")

        if expanded_ratio <= 0:
            raise ValueError("expanded_ratio must be positive")

        if n_encoder_layers < 1:
            raise ValueError("n_encoder_layers must be at least 1")
        if n_decoder_layers < 1:
            raise ValueError("n_decoder_layers must be at least 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.expanded_ratio = expanded_ratio
        self.type = type
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.expanded_dim = int(hidden_dim * expanded_ratio)

        # Build encoder: input -> ... -> bottleneck
        encoder_layers = []
        if n_encoder_layers == 1:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        else:
            # Compute intermediate dimensions (n_layers+1 elements)
            for i in range(n_encoder_layers):
                encoder_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                if i < n_encoder_layers - 1:  # Add ReLU between layers, not after last
                    encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Expansion layer: bottleneck -> expanded dimension
        self.expansion = nn.Linear(hidden_dim, self.expanded_dim, bias=False)
        # Build decoder: expanded_dim -> ... -> input_dim
        decoder_layers = []
        if n_decoder_layers == 1:
            decoder_layers.append(nn.Linear(self.expanded_dim, input_dim, bias=False))
        else:
            # Compute intermediate dimensions
            for i in range(n_decoder_layers - 1):
                decoder_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                if i < n_decoder_layers - 2:  # Don't add ReLU after last layer
                    decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)



    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the sparse autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - reconstructed: Reconstructed input tensor
            - features: Bottleneck features (before/after expansion based on type)
        """
        # Encode to bottleneck
        hidden = self.encoder(x)

        # Expand to larger dimension
        expanded = self.expansion(hidden)

        # Apply sparsity mechanism
        if self.type == "topk":
            # Top-k sparsity: keep only top k activations
            # Apply ReLU first
            expanded = F.relu(expanded)
            # Keep top k% of activations (default: top 10%)
            k = max(1, self.expanded_dim // 10)
            topk_values, topk_indices = torch.topk(expanded, k, dim=1)
            sparse_features = torch.zeros_like(expanded)
            sparse_features.scatter_(1, topk_indices, topk_values)
            features = sparse_features
        else:

            features = F.relu(expanded)

        # Decode back to input dimension
        reconstructed = self.decoder(features)

        return reconstructed, features

    def get_sparsity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss based on the type.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor from the forward pass.

        Returns
        -------
        torch.Tensor
            Sparsity loss value.
        """
        if self.type == "topk":
            # Top-k doesn't need explicit sparsity loss since it's enforced in forward
            return torch.tensor(0.0, device=features.device)
        else:  # vanilla
            # L1 regularization on features for vanilla
            return torch.mean(torch.abs(features))

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE).

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor.
        reconstructed : torch.Tensor
            Reconstructed tensor from the decoder.

        Returns
        -------
        torch.Tensor
            Reconstruction loss (MSE).
        """
        return F.mse_loss(reconstructed, x)

    def get_total_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor,
        sparsity_weight: float = 1e-3,
    ) -> torch.Tensor:
        """
        Compute total loss (reconstruction + sparsity).

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor.
        reconstructed : torch.Tensor
            Reconstructed tensor.
        features : torch.Tensor
            Feature tensor.
        sparsity_weight : float, default=1e-3
            Weight for sparsity loss.

        Returns
        -------
        torch.Tensor
            Total loss.
        """
        recon_loss = self.get_reconstruction_loss(x, reconstructed)
        sparsity_loss = self.get_sparsity_loss(features)
        return recon_loss + sparsity_weight * sparsity_loss

