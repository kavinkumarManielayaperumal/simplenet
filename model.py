import torch 
import torch.nn as nn# pytorch have the bulit- in neuaral network model , so that we are the using the directlly the pytorch model

class SimpleNet(torch.nn.Module):
    def __init__( self,input_size=10, hidden_size=10, output_size=1):
        super(SimpleNet,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.fc1=nn.Linear(self.input_size,self.hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(self.hidden_size,self.output_size)
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

if __name__=="__main__":
    model=SimpleNet() #in
    print(model)
    
    
    sample_data=torch.rand(5,10)
    output=model(sample_data)
    print(output)