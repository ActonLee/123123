from try_udfnet import MyDataset, test, train, visualise_prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.fields import UDFNetwork
from torch.utils.data import Dataset, DataLoader
from mesh2udf import getData
import os
from accelerate import Accelerator




accelerator = Accelerator()




if __name__ == "__main__":
  data_filepath =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  for folder in os.listdir(f"{data_filepath}/garment_dataset/shirt_dataset_rest"):
    mesh_fp = f"{data_filepath}/garment_dataset/shirt_dataset_rest/{folder}/shirt_mesh_r_clean.obj"
    file_name = folder


    shuffle_seed = 42
    np.random.seed(shuffle_seed)
    batch_size = 512
    learning_rate = 1e-3
    epochs = 150

    # control density of point-clouds
    # it's very important to adjust epsilon to match the measurement of the hard spaces of the mesh; 
    # and to adjust M to increase the density of pc around the hard spaces
    N = 50000
    epsilon = 0.035  #0.1
    M = 25
    
    netconf = {
        'd_out' : 1, 
        'd_in' : 3,
        'd_hidden' : 128,
        'n_layers' : 3,
        'skip_in' : (),
        'multires' : 6,
        'bias' : 0.5,  
        'scale' : 1.0,
        'udf_shift' : 0.0,   # udf intialization
        'geometric_init' : True,
        'weight_norm' : True,
        'udf_type' : 'abs',  # square or abs
        'predict_grad' : False,
      }

    X_train, y_train = getData(N, epsilon, M, mesh_fp, norm=True)  # both np.float32
    randomize = np.arange(len(X_train))
    np.random.shuffle(randomize)
    X_train = X_train[randomize]
    y_train = y_train[randomize]
    train_dataset = MyDataset(X_train, y_train)  # torch.float32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    UDFNet = UDFNetwork(**netconf)
    UDFNet.load_state_dict(torch.load("./N6BADMUAXYXO.pth"))
    
    Loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(UDFNet.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.2) 

    UDFNet, optimizer, train_dataloader, scheduler = accelerator.prepare(
        UDFNet, optimizer, train_dataloader, scheduler
    )

    for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train_loss = train(train_dataloader, UDFNet, Loss_fn, optimizer, accelerator)
      scheduler.step()
      if (t % 10 == 0 or t == epochs - 1):
        UDFNet.to("cpu")
        visualise_prediction(model=UDFNet, epoch=t, exportdir='.', name=file_name, resolution=100, dist_threshold_ratio=5.0)
        UDFNet.to("cuda")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(UDFNet)
    os.makedirs(os.path.join('.', f'model_dicts'), exist_ok=True)
    accelerator.save(unwrapped_model.state_dict(), f"./model_dicts/{file_name}.pth")


