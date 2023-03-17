import torch
import torch.nn as nn

# Define the neural network model
class SquaredModel(nn.Module):
    def __init__(self):
        super(SquaredModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred ** 2

# Initialize the model, loss function, and optimizer
model = SquaredModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model on a dataset of input-output pairs
train_data = [(torch.randn(1), torch.randn(1)**2) for _ in range(1000)]
for epoch in range(1000):
    for x, y_true in train_data:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

# Evaluate the model on a validation set
val_data = [(torch.randn(1), torch.randn(1)**2) for _ in range(100)]
total_loss = 0
with torch.no_grad():
    for x, y_true in val_data:
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        total_loss += loss.item()
mean_loss = total_loss / len(val_data)
print("Validation loss:", mean_loss)

# Allow the user to input a number to calculate the square power of
while True:
    try:
        x_test = float(input("Enter a number (or 'q' to quit): "))
    except ValueError:
        print("Please enter a valid number")
        continue
    if x_test == 'q':
        break
    x_tensor = torch.tensor([x_test])
    y_test = model(x_tensor)
    print("The squared value of", x_tensor.item(), "is", y_test.item())
