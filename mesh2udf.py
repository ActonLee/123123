import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize(v):
    centroid = np.mean(v, axis=0)
    v = v - centroid
    m = np.max(np.sqrt(np.sum(v ** 2, axis=1)))
    v = v / m
    return v

def visualize(pc, udf, dist):
    ''' 
    input: 
        pc: (N, 3) numpy array
        udf: (N, ) numpy array
        dist: float
            distance of the picked points
    ''' 
    # 筛选衣服mesh附近的点进行绘制
    pc = pc[udf < dist]  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.4)

    # 设置轴标签
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

def getData(N, epsilon, M, mesh_fp, norm=True):
    '''
    input: 
        N: number of original query points
        epsilon: range of neighbor points
        M: number of neighbor points
        mesh_fp: filepath of mesh
    
    return: 
        query_points: np (N, 3), 
        udf: np (N, )
        
    '''
    # read mesh && normalize vertices
    mesh = o3d.io.read_triangle_mesh(mesh_fp)
    if norm:
        v = np.asarray(mesh.vertices)  #np.float64
        v = normalize(v) 
        mesh.vertices = o3d.utility.Vector3dVector(v)
    target_points = np.asarray(mesh.vertices) # (N, 3)

    # get N query points
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    min_bound = np.array([-1., -1., -1.])
    max_bound = np.array([1., 1., 1.])
    #min_bound = mesh.vertex.positions.min(0).numpy()
    #max_bound = mesh.vertex.positions.max(0).numpy()
    #min_bound = min_bound - np.array([0.1, 0.1, 0.1])
    #max_bound = max_bound + np.array([0.1, 0.1, 0.1])
    query_points = np.random.uniform(low=min_bound, high=max_bound,
                                    size=[N, 3]).astype(np.float32)
    
    for point in target_points:
        e = np.array([epsilon, epsilon, epsilon])
        neighbor_points = np.random.uniform(low=point - e, high=point + e, 
                          size=[M, 3]).astype(np.float32)
        query_points = np.concatenate((query_points, neighbor_points), axis=0)
    
    udf = scene.compute_distance(query_points).numpy()
    #print(f"Normalized Mesh Min_bound: {min_bound}, \nNormalized Mesh Max_bound: {max_bound}")
    return (query_points, udf)


if __name__ == "__main__":
    N = 100000 # number of query points
    cur_filepath = os.path.dirname(os.path.abspath(__file__))
    mesh_fp = f"{cur_filepath}/skirt_4_panels_ZIV8FMYAW7_sim.obj"
    X, y = getData(N, epsilon=0.05, M = 20, mesh_fp=mesh_fp)
    print(type(X[0, 0]), type(y[0]))
    # Visualize query points
    visualize(X, y, dist=0.2)

