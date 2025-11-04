import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
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

if __name__ == '__main__':
    input_dim = 768
    hidden_dim = 768 * 8
    l1_lambda = 0.001
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 32
    model = SparseAutoencoder(d_in=input_dim, d_hidden=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        dummy_input = torch.randn(batch_size, input_dim)
        reconstructed_output, hidden_activations = model(dummy_input)
        recon_loss = F.mse_loss(reconstructed_output, dummy_input)
        l1_loss = l1_lambda * torch.linalg.vector_norm(hidden_activations, ord=1, dim=1).mean()
        total_loss = recon_loss + l1_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            avg_l0_norm = (hidden_activations > 0).float().sum(dim=1).mean().item()
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Total Loss: {total_loss.item():.6f}, "
                  f"Recon Loss: {recon_loss.item():.6f}, "
                  f"L1 Loss: {l1_loss.item():.6f}, "
                  f"Avg L0 (Active Features): {avg_l0_norm:.2f}")

