# import os
# from typing import Tuple, List
#
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
#
# class CombinedNN(nn.Module):
#     def __init__(self, input_channels=5, num_classes=1, grid_size=(10, 10), embedding_dim=16):
#         super(CombinedNN, self).__init__()
#         self.grid_size = grid_size
#         self.input_channels = input_channels
#
#         # Define the convolutional layers
#         self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16 + embedding_dim, 32, kernel_size=3, padding=1)  # Adjusted for embedding
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#
#         # Define the embedding layer for w
#         self.embedding = nn.Linear(1, embedding_dim)  # Replaced embedding layer with a linear transformation
#
#         # Compute the size of the flattened feature vector after the convolutions and pooling
#         conv_output_size = self._calculate_conv_output_size()
#         self.fc1 = nn.Linear(conv_output_size, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def _calculate_conv_output_size(self):
#         # Create a dummy tensor to pass through the convolutional layers to compute the output size
#         dummy_input = torch.zeros(1, self.input_channels, *self.grid_size)
#         x = F.relu(self.conv1(dummy_input))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#
#         # Embedding dimension needs to be accounted for here
#         dummy_embedding = torch.zeros(1, 16, x.size(2), x.size(3))  # Same dimension as the embedding_dim
#         x = torch.cat((x, dummy_embedding), dim=1)
#
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#         return x.numel()  # Number of elements in the flattened tensor
#
#     def forward(self, x, w):
#         # Apply the first convolution and max pooling
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#
#         # Transform w using a linear layer instead of an embedding
#         # Convert integer w to a float tensor
#         if isinstance(w, int):
#             w = torch.tensor(w, dtype=torch.float32).unsqueeze(0)  # Convert w to a tensor and add batch dimension
#         elif isinstance(w, torch.Tensor):
#             w = w.float()  # Ensure it's a float tensor
#             w = w.view(-1, 1)  # Reshape for linear layer if it's already a tensor
#         # Ensure w is treated as a float tensor
#         w_embedding = self.embedding(w).view(-1, 16, 1, 1)  # Transform and reshape
#         w_embedding = w_embedding.repeat(1, 1, x.size(2), x.size(3))  # Repeat to match spatial size of x
#
#         # Concatenate the transformed w with the output of conv1
#         x = torch.cat((x, w_embedding), dim=1)
#
#         # Continue through the rest of the network
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, kernel_size=2, stride=2)
#
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# class ModelManager:
#     def __init__(self, base_directory="models"):
#         self.base_directory = base_directory
#         # self.models = {}
#         pass
#
#     # def get_or_create_model(self, name, input_channels, num_classes=1):
#     #     if name not in self.models:
#     #         self.models[name] = CombinedNN(input_channels, num_classes)
#     #     return self.models[name]
#
#     @staticmethod
#     def train_model(model, grid_inputs, w_inputs, labels, epochs=100, learning_rate=0.0001, model_name="model"):
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#         losses = []
#         for epoch in range(epochs):
#             model.train()
#             optimizer.zero_grad()
#             # output = model(grid_inputs,w_inputs)
#             # total_loss = criterion(output,labels)
#             total_loss = 0.0
#             for i in range(len(grid_inputs)):
#                 output = model.forward(grid_inputs[i], w_inputs[i])
#                 loss = criterion(output.view(-1), torch.tensor(labels[i], dtype=torch.float32).view(-1))
#                 total_loss += loss.item()
#                 loss.backward()
#                 optimizer.step()
#                 print(f'\tW: {w_inputs[i]}, Path Length Label: {labels[i]}, Model Output: {output.item()}, Loss: {loss.item()}')
#             print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}')
#             losses.append(total_loss)
#
#         print('Training complete')
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, epochs + 1), losses, label="Training Loss")
#         plt.xlabel("Epoch")
#         plt.ylabel("Loss")
#         plt.title("Training Loss Over Epochs")
#         plt.yscale('log')  # Logarithmic scale due to potentially large loss values
#         plt.legend()
#         plt.grid(True)
#         directory = os.path.dirname(model_name)
#         os.makedirs(directory, exist_ok=True)
#         plt.savefig(f'{model_name}_loss.png')
#         # plt.show()
#
#     def save_model(self, model, problem_name, model_name="model.pth"):
#         directory = os.path.join(self.base_directory, problem_name)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         model_path = os.path.join(directory, model_name)
#         torch.save(model.state_dict(), model_path)
#
#     def get_model_path(self, problem_name, model_name="model.pth"):
#         return os.path.join(self.base_directory, problem_name, model_name)
#
#     def load_model(self, problem_name,
#                    model_class=CombinedNN,
#                    model_name="model.pth",
#                    *args, **kwargs):
#         model_instance = model_class(*args, **kwargs)
#         model_path = self.get_model_path(problem_name, model_name)
#         if os.path.exists(model_path):
#             state_dict = torch.load(model_path)
#             model_instance.load_state_dict(state_dict)
#             model_instance.eval()
#             return model_instance
#         else:
#             print(f"No model found at {model_path}")
#             return None
#
#     def get_or_create_model(self, problem_name, model_class=CombinedNN, *args, **kwargs):
#         model = self.load_model(problem_name, model_class, *args, **kwargs)
#         if model is None:
#             model = model_class(*args, **kwargs)
#         return model


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple


class GridStateCNN(nn.Module):
    def __init__(self, grid_size: Tuple[int, int], num_channels: int = 5):
        super(GridStateCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        conv_output_size = self._calculate_conv_output_size(grid_size)

        # Separate layer to process T
        self.fc_T = nn.Linear(in_features=1, out_features=64)

        # Combined layer after concatenation
        self.fc1 = nn.Linear(in_features=conv_output_size + 64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def _calculate_conv_output_size(self, grid_size: Tuple[int, int]):
        dummy_input = torch.zeros(1, 5, *grid_size)
        x = F.relu(self.conv1(dummy_input))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x.numel()

    def forward(self, grid_tensor: torch.Tensor, T: int):
        # Example scaling: make T more comparable to the grid size

        # Convert to tensor
        T_tensor = torch.tensor([T], dtype=torch.float32).unsqueeze(0)

        # Pass through fully connected layer for T
        T_out = F.relu(self.fc_T(T_tensor))

        # Pass through convolutional layers
        x = F.relu(self.conv1(grid_tensor))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        # Concatenate processed T with the flattened grid features
        x = torch.cat((x, T_out), dim=1)

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

    def train_model(self, inputs: List[torch.Tensor], t_inputs: List[int], epochs: int, labels: List[int]):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(inputs)):
                grid_tensor = inputs[i]
                t_input = t_inputs[i]
                label = torch.tensor([labels[i]], dtype=torch.float32).unsqueeze(0)

                if grid_tensor.dim() == 3:
                    grid_tensor = grid_tensor.unsqueeze(0)
                    grid_tensor = grid_tensor.permute(0, 3, 1, 2)

                output = self.forward(grid_tensor, t_input)

                if output.dim() == 1:
                    output = output.unsqueeze(1)

                loss = criterion(output, label)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

