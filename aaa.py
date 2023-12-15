import torch
from extract_mesh import get_mesh_udf_fast
import trimesh
import os

def extract_udf_mesh(model, epoch, exportdir, name, world_space=False, resolution=256, dist_threshold_ratio=1.0):
    func = model.udf
    def func_grad(xyz):
        gradients = model.gradient(xyz)
        gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
        gradients_norm = gradients / (gradients_mag + 1e-5)  # normalize to unit vector
        return gradients_norm

    pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                    indices=None, N_MC=resolution,
                                                                    gradient=True, eps=0.005,
                                                                    border_gradients=True,
                                                                    smooth_borders=True,
                                                                    dist_threshold_ratio=dist_threshold_ratio)

    vertices, triangles = pred_mesh.vertices, pred_mesh.faces
    #if world_space:
    #    vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

    mesh = trimesh.Trimesh(vertices, triangles)

    os.makedirs(os.path.join(exportdir, f'udf_meshes'), exist_ok=True)
    mesh.export(
        os.path.join(exportdir, f'udf_meshes', 'udf_fn{}_res{}_step{}.ply'.format(name, resolution, epoch)))