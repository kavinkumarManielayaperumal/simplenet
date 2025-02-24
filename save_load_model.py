import torch
from model import SimpleNet

# Save the model
model = SimpleNet()
# ... train the model ...
torch.save(model.state_dict(), 'model.pth')

# Load the model
loaded_model = SimpleNet()
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

