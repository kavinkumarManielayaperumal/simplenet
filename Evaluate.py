import torch
from model import SimpleNet

def evaluate_model(filename='model.pth'):
    device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
    model=SimpleNet().to(device)
    print(f"model is running on the device:{device}")
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model
model=evaluate_model()
test_data=torch.rand(5,10).to(model.device)
prediction=model(test_data)
print(prediction)