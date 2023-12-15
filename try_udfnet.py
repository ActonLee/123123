import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.fields import UDFNetwork
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mesh2udf import getData
import os
from aaa import extract_udf_mesh
import visdom


class MyDataset(Dataset):
  def __init__(self, X, y):
    '''
    X: np (N, 3)
    y: np (N,)
    '''
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)
    
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, index):
    '''
    output: torch tensor ((3, ), (1, ))
    '''
    return (self.X[index], self.y[index])


def train(dataloader, model, loss_fn, optimizer, accelerator):
  num_batches = len(dataloader)
  model.train()
  size = len(dataloader.dataset)
  train_loss = 0

  for batch, (X, y) in enumerate(dataloader):
    optimizer.zero_grad()
    
    pred = model.udf(X).squeeze()
    
    loss = loss_fn(pred, y)
    train_loss += loss.item()

    accelerator.backward(loss)
    optimizer.step()
    
    if batch % 50 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss * 10000 :>12f}  [{current:>5d}/{size:>5d}]")
  
  train_loss /= num_batches
  print(f"Average Train Loss: {train_loss * 10000 :>12f}")   
  return train_loss * 10000 
  

def test(dataloader, model, loss_fn):
  num_batches = len(dataloader)
  model.eval()
  test_loss = 0
  with torch.no_grad():
      for X, y in dataloader:
          pred = model.udf(X).squeeze()
          test_loss += loss_fn(pred, y).item()
  test_loss /= num_batches
  print(f"Test Error: \n Avg loss: {test_loss * 10000 :>12f} \n")


def visualise_prediction(model, epoch, exportdir, name, resolution=128, dist_threshold_ratio=1.0):
  model.eval()
  extract_udf_mesh(resolution=resolution, model=model, epoch=epoch, exportdir=exportdir, name=name, world_space=True, dist_threshold_ratio=dist_threshold_ratio)
  

if __name__ == "__main__":
  
  cur_filepath = os.path.dirname(os.path.abspath(__file__))
  mesh_fp = f"{cur_filepath}/skirt_4_panels_ZIV8FMYAW7_sim.obj"
  file_name = "skirt_4_panels_ZIV8FMYAW7"
  
  #test_size = 1
  shuffle_seed = 42
  np.random.seed(shuffle_seed)
  batch_size = 512
  learning_rate = 1e-3
  epochs = 75

  # control density of point-clouds
  # it's very important to adjust epsilon to match the measurement of the hard spaces of the mesh; 
  # and to adjust M to increase the density of pc around the hard spaces
  N = 50000
  epsilon = 0.12
  M = 50
  
  netconf = {
      'd_out' : 257, 
      'd_in' : 3,
      'd_hidden' : 256,
      'n_layers' : 8,
      'skip_in' : (4,),
      'multires' : 6,
      'bias' : 0.5,  
      'scale' : 1.0,
      'udf_shift' : 0.0,   # udf intialization
      'geometric_init' : True,
      'weight_norm' : True,
      'udf_type' : 'abs',  # square or abs
      'predict_grad' : False,
    }
  
  #Visualize Losses
  #vis = visdom.Visdom(env='train_udfnet')
  #vis.line([0.], # Y的第一个点的坐标
	#	  [0.], # X的第一个点的坐标
	#	  win = '1', # 窗口的名称
	#	  opts = dict(title = 'train_loss')) # 图像的标例
    
  #X, y = getData(N, mesh_fp=mesh_fp, norm=True)
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=split_seed, shuffle=True)
  X_train, y_train = getData(N, epsilon, M, mesh_fp, norm=True)
  randomize = np.arange(len(X_train))
  np.random.shuffle(randomize)
  X_train = X_train[randomize]
  y_train = y_train[randomize]
  train_dataset = MyDataset(X_train, y_train)
  #test_dataset = MyDataset(X_test, y_test)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
  #test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

  device = (
      "cuda"
      if torch.cuda.is_available()
      else "mps"
      if torch.backends.mps.is_available()
      else "cpu"
  )
  print(f"Using {device} device")
  if (device != "cuda"):
    assert 0==1
  UDFNet = UDFNetwork(**netconf).to(device)
  print(UDFNet)

  Loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(UDFNet.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 65, 85, 90, 95, 100], gamma=0.5) 

  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, UDFNet, Loss_fn, optimizer)
    scheduler.step()
    print(scheduler.get_last_lr())
    
    #vis.line([train_loss], [t + 1], win='1', update='append')
    #test(test_dataloader, UDFNet, Loss_fn)

    if (t % 10 == 0 or t == epochs - 1):
      UDFNet.to("cpu")
      visualise_prediction(model=UDFNet, epoch=t, exportdir='.', name=file_name, resolution=100, dist_threshold_ratio=5.0)
      UDFNet.to(device)

    
    
    