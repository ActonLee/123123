import torch
from models.fields import UDFNetwork
from try_udfnet import visualise_prediction


if __name__ == "__main__":
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
    
    udfnet = UDFNetwork(**netconf)
    udfnet.load_state_dict(torch.load("./model_dicts/A3YFCMWTQBDM.pth", map_location=torch.device("cpu")))
    visualise_prediction(model=udfnet, epoch=0, exportdir='.', name="A3YFCMWTQBDM", resolution=100, dist_threshold_ratio=5.0)