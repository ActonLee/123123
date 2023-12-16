# 123123

### TASK:
input: mesh * 7520

output: udf   --marching_cubes-->  reconstructed mesh 

pipline:

1. use functions in mesh2udf.py to sample point cloud on a mesh and paired with udf value

2. feed an initialised udfnet with pc-udf pair as its input and output, train

3. finish training. save the model_dict of the trained udfnet, which is what we need

4. repeat  1, 2, 3 on a new mesh and an initialised udfnet. 

5. remember: each mesh is paired with a udfnet model dict, saved in `./model_dicts`




---

### FILES:

#### run_udfnet.py
1. use command of accelerate to run this py file
2. start the whole train process

#### try_udfnet.py
1. contain functions like `train`, `visualise_prediction` and class `MyDataset` used in run_udfnet.py

#### mesh2udf.py
1. utils used to sample pc and paired udf value


---
### DATASET:
garment_dataset:
contain 7520 folders, each with a mesh in it. 
