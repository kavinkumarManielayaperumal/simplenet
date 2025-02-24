
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNet
from data_loader import get_dataloader

def train_model(model, dataloader,num_epochs=10, lr=0.01):
    criterion = nn.MSELoss() # mean squared error loss for the regression proble, if we have the classification probolem we will use the cross entropy loss function
    optimizer= optim.SGD(model.parameters(), lr=lr) # the parameters are the weights and biases of the model
    
    
    for epoch in range(num_epochs):#in
        total_loss=0.0
        for batch_data, batch_lables in dataloader: # this will give the data and lables in the dataloader  
            optimizer.zero_grad() # this is used to make the gradient zero after each epoch
            
            outputs= model(batch_data)# this main fuction this go into the model.py file and perdict the output without the lables
            loss=criterion(outputs, batch_lables) # this caluclate the loss between the predicted output
            loss.backward() # update the weights and biases of the model but this inbuile function come from the pytorch ,if you want know more about this backward function you can see the pytorch doucmenta
            optimizer.step()
            
            total_loss+=loss.item()#in
        print(f"Epoch:[{epoch+1}/{num_epochs}, losss:{total_loss/len(dataloader)}")
    save_model(model, filename='model.pth') # this child function is used to save the model, so now calling this function in parent function
    
def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(),filename)# this in built funciton in the pytorch to saave the model in the
    print("Model saved successfully")
    
    # if is stored in the dictionary formate , so that we can see the weights and biases of the model
    print(f" weight and bias{model.state_dict()}")
    loaded_model=SimpleNet() 
    loaded_model.load_state_dict(torch.load(filename)) # whenever try to use the model for different purpose without training from first insted of we can use the learned parameters
    loaded_model.eval()

if __name__=="__main__":#
    dataloader=get_dataloader(csv_file='dataset.csv',  batch_size=32)
    model=SimpleNet()
    
    train_model(model, dataloader,num_epochs=10, lr=0.01)
    