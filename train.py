from networks.pinn import PINN  
import torch.optim as optim
from loss.pinn_loss import loss_function


def train_PINN(x_tensor, t_tensor, u_tensor,**params): 

    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5000

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_function(model, x_tensor, t_tensor, u_tensor,**params)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
