import argparse
import numpy as np
from skimage import measure
from sklearn.neighbors import KDTree
import open3d as o3d;

def createGrid(points, resolution=96):
    """
    constructs a 3D grid containing the point cloud
    each grid point will store the implicit function value
    Args:
        points: 3D points of the point cloud
        resolution: grid resolution i.e., grid will be NxNxN where N=resolution
                    set N=16 for quick debugging, use *N=64* for reporting results
    Returns: 
        X,Y,Z coordinates of grid vertices     
        max and min dimensions of the bounding box of the point cloud                 
    """
    max_dimensions = np.max(points,axis=0) # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points,axis=0) # smallest x, smallest y, smallest z coordinates among all surface points    
    bounding_box_dimensions = max_dimensions - min_dimensions # com6pute the bounding box dimensions of the point cloud
    max_dimensions = max_dimensions + bounding_box_dimensions/10  # extend bounding box to fit surface (if it slightly extends beyond the point cloud)
    min_dimensions = min_dimensions - bounding_box_dimensions/10
    X, Y, Z = np.meshgrid( np.linspace(min_dimensions[0], max_dimensions[0], resolution),
                           np.linspace(min_dimensions[1], max_dimensions[1], resolution),
                           np.linspace(min_dimensions[2], max_dimensions[2], resolution) )    
    
    return X, Y, Z, max_dimensions, min_dimensions

def sphere(center, R, X, Y, Z):
    """
    constructs an implicit function of a sphere sampled at grid coordinates X,Y,Z
    Args:
        center: 3D location of the sphere center
        R     : radius of the sphere
        X,Y,Z : coordinates of grid vertices                      
    Returns: 
        IF    : implicit function of the sphere sampled at the grid points
    """    
    IF = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2 - R ** 2 
    return IF

def showMeshReconstruction(IF):
    """
    calls marching cubes on the input implicit function sampled in the 3D grid
    and shows the reconstruction mesh
    Args:
        IF    : implicit function sampled at the grid points
    """    
    verts, triangles, normals, values = measure.marching_cubes(IF, 0)        

    # Create an empty triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    # Use mesh.vertex to access the vertices' attributes    
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    # Use mesh.triangle to access the triangles' attributes    
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()        
    o3d.visualization.draw_geometries([mesh]) 

def mlsReconstruction(points, normals, X, Y, Z, K=50):
    """
    Surface reconstruction with an implicit function f(x,y,z) representing
    MLS distance to the tangent plane of the input surface points.
    Args:
        points :  points of the point cloud
        normals:  normals of the point cloud
        X,Y,Z  :  coordinates of grid vertices 
        k      :  Number of nearest neighbors to consider (default: 50)
    Returns:
        IF     :  implicit function sampled at the grid points
    """

    IF = np.random.rand(X.shape[0], X.shape[1], X.shape[2]) - 0.5

    tree = KDTree(points)

    beta = 2 * np.mean(np.max(tree.query(points, k=2)[0][:, 1:], axis=1))

    grid_coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    _, idx = tree.query(grid_coords, k=K)
    closest_points = points[idx]
    closest_normals = normals[idx]
    k_distances = np.linalg.norm(closest_points - grid_coords[:, None], axis=-1)
    phi = np.exp(-k_distances**2 / beta**2)
    phi_distances = np.sum(phi, axis=-1)
    phi_normals = phi[..., None] * closest_normals

    IF = np.sum(np.sum(phi_normals * (grid_coords[:, None, :] - closest_points), axis=-1), axis=-1)/ phi_distances

    IF = IF.reshape(X.shape)

    return IF



def naiveReconstruction(points, normals, X, Y, Z):
    """
    Surface reconstruction with an implicit function f(x,y,z) representing
    signed distance to the tangent plane of the surface point nearest to each 
    point (x,y,z).
    Args:
        points :  points of the point cloud
        normals:  normals of the point cloud
        X,Y,Z  :  coordinates of grid vertices 
    Returns:
        IF     : implicit function sampled at the grid points
    """

    IF = np.zeros_like(X)  # Initialize implicit function with zeros

    # Construct KDTree for efficient nearest neighbor search
    tree = KDTree(points)

    # For each grid point, find the nearest surface point and compute signed distance
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Z.shape[2]):
                query_point = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                _, idx = tree.query(query_point.reshape(1, -1), k=1)  # Find the nearest neighbor
                nearest_point = points[idx[0][0]]
                normal = normals[idx[0][0]]
                # Compute signed distance using dot product with normal
                IF[i, j, k] = np.dot(normal, query_point - nearest_point)

    return IF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic surface reconstruction')
    parser.add_argument('--file', type=str, default = "sphere.pts", help='input point cloud filename')
    parser.add_argument('--method', type=str, default = "sphere",\
                        help='method to use: mls (Moving Least Squares), naive (naive reconstruction), sphere (just shows a sphere)')
    args = parser.parse_args()

    #load the point cloud
    data = np.loadtxt(args.file)
    points = data[:, :3]
    normals = data[:, 3:6]

    # create grid whose vertices will be used to sample the implicit function
    X,Y,Z,max_dimensions,min_dimensions = createGrid(points, 64)

    if args.method == 'mls':
        print(f'Running Moving Least Squares reconstruction on {args.file}')
        IF = mlsReconstruction(points, normals, X, Y, Z)
    elif args.method == 'naive':
        print(f'Running naive reconstruction on {args.file}')
        IF = naiveReconstruction(points, normals, X, Y, Z)
    else:
        # toy implicit function of a sphere - replace this code with the correct
        # implicit function based on your input point cloud!!!
        print(f'Replacing point cloud {args.file} with a sphere!')
        center =  (max_dimensions + min_dimensions) / 2
        R = max( max_dimensions - min_dimensions ) / 4
        IF =  sphere(center, R, X, Y, Z)

    showMeshReconstruction(IF)
