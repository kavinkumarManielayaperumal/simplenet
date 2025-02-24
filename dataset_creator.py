import numpy as np 
import pandas as pd
import torch

def creation_of_dataset(num_samples=100,num_features=10):
    np.random.seed(42)
    data=np.random.rand(num_samples,num_features).astype(np.float32)
    labels=np.random.rand(num_samples,1).astype(np.float32)
    
    # if i want independent data and lables i will use bleow code # if you change the NumPy array,the PyTorch tensor will not change automatically.
    data_torch=torch.tensor(data)
    labels_torch=torch.tensor(labels) # cons of this is that it will not be able to use the numpy functions like np.random.rand
    
    #if i want to use the same data and labels i will use below code # If you change the NumPy array, the PyTorch tensor also changes automatically beacuse they have the from_numpy function
    #data_torch=torch.from_numpy(data)
    #labels_torch=torch.from_numpy(labels) #cons of this is that it will not be able to use the torch functions like torch.rand
    
    # for the stroing the data and labels in the csv file we need allways numpy array beacuse troch are not supported in the csv file ** important
    
    df=pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(num_features)])
    df['Label']=labels
    
    df.to_csv('dataset.csv',index=False)
    
    print("Dataset created successfully")
    return data_torch, labels_torch

 # its simply act like pass key word , so that we can use this function in the main.py file 
if __name__=='__main__':#If you create a new file called main.py and want to use dataset_creator.py inside it, the if __name__ == "__main__": check ensures that create_dataset() does not run automatically when imported.
	creation_of_dataset()