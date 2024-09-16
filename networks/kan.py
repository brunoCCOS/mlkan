
import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, Bs):
        super(CustomLayer, self).__init__()
        self.Bs = Bs  # List of basis functions
        self.K = len(Bs)  # Number of basis functions
        self.c = nn.Parameter(torch.randn(self.K, out_features))  # Trainable parameters c_k
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x shape: (batch_size, in_features)
        batch_size = x.size(0)
        S = []
        for B_k in self.Bs:
            Bx = B_k(x)  # Apply basis function to x
            S_k = Bx.sum(dim=1, keepdim=True)  # Sum over input features
            S.append(S_k)  # Collect sums for each basis function
        S = torch.cat(S, dim=1)  # Shape: (batch_size, K)
        output = S @ self.c  # Matrix multiplication with trainable parameters
        return output  # Shape: (batch_size, out_features)
class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_terms=None):
        """
        Initialize the Kolmogorov–Arnold Network.

        Parameters:
        - input_dim (int): Dimension of the input data.
        - hidden_dim (int): Number of neurons in the hidden layers of univariate functions.
        - output_dim (int): Dimension of the output data.
        - num_terms (int): Number of terms in the summation (default is 2*input_dim + 1).
        """
        super(KolmogorovArnoldNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_terms = num_terms if num_terms is not None else 2 * input_dim + 1

        # Define the ψ_{pq} functions: one for each input dimension and summation term
        self.psi_functions = nn.ModuleList([
            nn.ModuleList([
                self._build_univariate_function(hidden_dim)
                for _ in range(self.num_terms)
            ]) for _ in range(self.input_dim)
        ])

        # Define the Φ_q functions: one for each summation term and output dimension
        self.phi_functions = nn.ModuleList([
            nn.ModuleList([
                self._build_univariate_function(hidden_dim)
                for _ in range(self.output_dim)
            ]) for _ in range(self.num_terms)
        ])

    def _build_univariate_function(self, hidden_dim):
        """
        Build a univariate function modeled by a small neural network.

        Parameters:
        - hidden_dim (int): Number of neurons in the hidden layers.

        Returns:
        - nn.Sequential: The univariate function as a neural network.
        """
        return nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        batch_size = x.size(0)

        # Compute s_q = ∑_{p=1}^{n} ψ_{pq}(x_p)
        s_q = []
        for q in range(self.num_terms):
            sum_p = torch.zeros(batch_size, 1, device=x.device)
            for p in range(self.input_dim):
                x_p = x[:, p].unsqueeze(1)  # Shape: (batch_size, 1)
                psi_pq = self.psi_functions[p][q](x_p)  # Univariate function
                sum_p += psi_pq
            s_q.append(sum_p)  # s_q[q] has shape (batch_size, 1)

        # Compute the output: f_i(x) = ∑_{q=0}^{2n} Φ_q(s_q)
        outputs = []
        for i in range(self.output_dim):
            f_i = torch.zeros(batch_size, 1, device=x.device)
            for q in range(self.num_terms):
                phi_qi = self.phi_functions[q][i](s_q[q])  # Univariate function
                f_i += phi_qi
            outputs.append(f_i)

        # Concatenate outputs for each output dimension
        output = torch.cat(outputs, dim=1)  # Shape: (batch_size, output_dim)
        return output

# Example usage
if __name__ == "__main__":
    # Define network parameters
    input_dim = 3
    hidden_dim = 10
    output_dim = 1
    num_terms = 2 * input_dim + 1

    # Instantiate the network
    model = KolmogorovArnoldNetwork(input_dim, hidden_dim, output_dim, num_terms)

    # Sample input
    x = torch.randn(5, input_dim)  # Batch of 5 samples

    # Forward pass
    output = model(x)

    print("Output shape:", output.shape)  # Should be (5, output_dim)
    print("Output:", output)
