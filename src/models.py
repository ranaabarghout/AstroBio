"""
Sparse Autoencoder (SAE) implementation.

Supports two types:
- vanilla: Uses L1 regularization for sparsity
- topk: Uses top-k activation sparsity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

class SimpleSparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden):
        super(SparseAutoencoder, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.encoder = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(d_hidden, d_in)

    def forward(self, z):
        hidden_activations = self.relu(self.encoder(z))
        reconstructed_z = self.decoder(hidden_activations)
        return reconstructed_z, hidden_activations


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
        return recon_loss, sparsity_loss, recon_loss + sparsity_weight * sparsity_loss


    def train_model(self, train_loader: DataLoader, test_loader: DataLoader, n_epochs: int = 1, lr: float = 1e-3, sparsity_weight: float = 1e-2, 
                    device: torch.device | None = None, plot_losses: bool = True) -> dict:
        """
        Train the sparse autoencoder model and track training/test losses per epoch.
        
        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training data loader.
        test_loader : torch.utils.data.DataLoader
            Test data loader.
        n_epochs : int, default=1
            Number of training epochs.
        lr : float, default=1e-3
            Learning rate for optimizer.
        sparsity_weight : float, default=1e-2
            Weight for sparsity loss.
        device : torch.device, optional
            Device to train on. If None, will use CUDA if available, else CPU.
        plot_losses : bool, default=True
            Plot losses per epoch.
        
        Returns
        -------
        dict
            Dictionary containing lists of epoch-level mean losses:
            - train_recon_loss: Mean training reconstruction losses per epoch
            - train_sparsity_loss: Mean training sparsity losses per epoch
            - train_total_loss: Mean training total losses per epoch
            - test_recon_loss: Mean test reconstruction losses per epoch
            - test_sparsity_loss: Mean test sparsity losses per epoch
            - test_total_loss: Mean test total losses per epoch
            - epochs: Epoch numbers
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        optimizer = Adam(self.parameters(), lr=lr)
        self.to(device)
        
        # Initialize epoch-level loss tracking lists
        train_recon_losses = []
        train_sparsity_losses = []
        train_total_losses = []
        test_recon_losses = []
        test_sparsity_losses = []
        test_total_losses = []
        epochs = []
        
        self.train()
        for i_epoch in range(n_epochs):
            # Track losses for this epoch
            epoch_train_recon = []
            epoch_train_sparsity = []
            epoch_train_total = []
            
            # Training phase
            for i_step, d in enumerate(train_loader):
                d = d.to(device)
                optimizer.zero_grad()
                recon, embed = self(d)
                recon_loss, sparsity_loss, total_loss = self.get_total_loss(
                    d, recon, embed, sparsity_weight=sparsity_weight
                )
                total_loss.backward()
                optimizer.step()
                
                # Collect training losses for this epoch
                epoch_train_recon.append(recon_loss.item())
                epoch_train_sparsity.append(sparsity_loss.item())
                epoch_train_total.append(total_loss.item())
                
            # Compute mean training losses for this epoch
            avg_train_recon = sum(epoch_train_recon) / len(epoch_train_recon)
            avg_train_sparsity = sum(epoch_train_sparsity) / len(epoch_train_sparsity)
            avg_train_total = sum(epoch_train_total) / len(epoch_train_total)
            
            train_recon_losses.append(avg_train_recon)
            train_sparsity_losses.append(avg_train_sparsity)
            train_total_losses.append(avg_train_total)
            
            # Evaluate on test set after each epoch
            self.eval()
            test_recon_loss_epoch = []
            test_sparsity_loss_epoch = []
            test_total_loss_epoch = []
            
            with torch.no_grad():
                for test_d in test_loader:
                    test_d = test_d.to(device)
                    test_recon, test_embed = self(test_d)
                    test_recon_loss, test_sparsity_loss, test_total_loss = self.get_total_loss(
                        test_d, test_recon, test_embed, sparsity_weight=sparsity_weight
                    )
                    test_recon_loss_epoch.append(test_recon_loss.item())
                    test_sparsity_loss_epoch.append(test_sparsity_loss.item())
                    test_total_loss_epoch.append(test_total_loss.item())
            
            # Average test losses for this epoch
            avg_test_recon = sum(test_recon_loss_epoch) / len(test_recon_loss_epoch)
            avg_test_sparsity = sum(test_sparsity_loss_epoch) / len(test_sparsity_loss_epoch)
            avg_test_total = sum(test_total_loss_epoch) / len(test_total_loss_epoch)
            
            test_recon_losses.append(avg_test_recon)
            test_sparsity_losses.append(avg_test_sparsity)
            test_total_losses.append(avg_test_total)
            epochs.append(i_epoch)
            
            print(f"\nEpoch {i_epoch} Summary:")
            print(f"  Train - Total: {avg_train_total:.4f}, Recon: {avg_train_recon:.4f}, Sparsity: {avg_train_sparsity:.4f}")
            print(f"  Test  - Total: {avg_test_total:.4f}, Recon: {avg_test_recon:.4f}, Sparsity: {avg_test_sparsity:.4f}\n")
            
            self.train()
        
        # Plot losses per epoch
        if plot_losses:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Plot reconstruction loss
            axes[0].plot(epochs, train_recon_losses, label='Train Recon Loss', marker='o', markersize=6)
            axes[0].plot(epochs, test_recon_losses, label='Test Recon Loss', marker='s', markersize=6)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Reconstruction Loss')
            axes[0].set_title('Reconstruction Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot sparsity loss
            axes[1].plot(epochs, train_sparsity_losses, label='Train Sparsity Loss', marker='o', markersize=6)
            axes[1].plot(epochs, test_sparsity_losses, label='Test Sparsity Loss', marker='s', markersize=6)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Sparsity Loss')
            axes[1].set_title('Sparsity Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot total loss
            axes[2].plot(epochs, train_total_losses, label='Train Total Loss', marker='o', markersize=6)
            axes[2].plot(epochs, test_total_losses, label='Test Total Loss', marker='s', markersize=6)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Total Loss')
            axes[2].set_title('Total Loss')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        
        return {
            'train_recon_loss': train_recon_losses,
            'train_sparsity_loss': train_sparsity_losses,
            'train_total_loss': train_total_losses,
            'test_recon_loss': test_recon_losses,
            'test_sparsity_loss': test_sparsity_losses,
            'test_total_loss': test_total_losses,
            'epochs': epochs
        }

