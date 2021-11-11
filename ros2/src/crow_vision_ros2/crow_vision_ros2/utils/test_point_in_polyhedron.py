from scipy.spatial import Delaunay
from crow_vision_ros2.utils.polyhedron import Polyhedron
import open3d as o3d

def get_triangles(mesh):
    return np.asarray(mesh.triangles)

def get_vertices(mesh):
    return np.asarray(mesh.vertices)

def get_mesh_from_points(points):
    alpha = 0.8
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([cloud])
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(cloud)
    #o3d.visualization.draw_geometries([tetra_mesh])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    #o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

def get_bbox_from_points(points):
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
    #o3d.visualization.draw_geometries([bbox])
    return bbox

def test_in_hull(p, points):
    """Test if point p is in hull defined by points, using Delaunay triangulation
    Args:
        p (list): tested point, 3d
        points (list of lists): polyhedron points, 3d
    Returns:
        result (bool): Whether or not the point is inside polyhedron
    """
    if not isinstance(points,Delaunay):
        hull = Delaunay(points)
    return hull.find_simplex(p)>=0

def test_in_mesh(p, points):
    """Test if point is in mesh defined by triangles and vertices
    Args:
        p (list): tested point, 3d
        points (list of lists): polyhedron points, 3d
    Returns:
        result (bool): Whether or not the point is inside polyhedron
    """
    mesh = get_mesh_from_points(polyhedron)
    #bbox = get_bbox_from_points(polyhedron)
    triangles = get_triangles(mesh)
    vertices = get_vertices(mesh)

    tetrahedron = Polyhedron(vertex_positions=np.asarray(vertices),triangles=np.asarray(triangles))
    answer = tetrahedron.winding_number(np.asarray(p))
    return answer
