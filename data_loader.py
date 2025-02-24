import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from dataset_creator import creation_of_dataset

class MyDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()# this is used to call the __init__ of the parent class , whenever first parent class is called then the child class is called 
        df=pd.read_csv(csv_file)
        #iloc is like the sclicing of the data frame, so there are loc and iloc loc is used for the lables and iloc is used for the data
        self.data=torch.tensor(df.iloc[:,:-1].values, dtype=torch.float32)
        self.lables=torch.tensor(df.iloc[:,:-1].values, dtype=torch.float32).reshape(-1,1)# reshaping the lables is beacuse we need in the column formate , if we dont reshape it will be in the row formate 

     # this both function are used to get the length of the data and get the data and lables like idexing the data and lables
    def __len__(self): 
        return len(self.data)


    def __getitem__(self,idx):
        return self.data[idx], self.lables[idx] #this will return the data and lables in the form of tuples with matchind index , so that lables and data will nor be missmatch


def get_dataloader(csv_file='dataset.csv', batch_size=32): # this function is used to get the data and lables in the form of batches
    dataset = MyDataset(csv_file)# calling the parent class MyDataset
    dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__=='__main__':

    creation_of_dataset()

    datafile = r"E:\for practice game\simplenet\dataset.csv"

    dataloader=get_dataloader(datafile, batch_size=32)

    for batch_data, batch_lables in dataloader:
        print("Batch Data:", batch_data)
        print("Batch Lables:", batch_lables)
        