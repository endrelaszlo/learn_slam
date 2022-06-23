import numpy as np
import open3d as o3d
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
# o3d.visualization.draw(mesh, raw_mode=True)

points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
          [0, 1, 1], [1, 1, 1]]
lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)


#  Source: https://ksimek.github.io/2013/08/13/intrinsic/
#  if you know the camera's film (or digital sensor) has a width W in millimiters, and the image width in pixels is w,
#  you can convert the focal length fx to world units using
#  Fx = fx * (W/w)

cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px=10,
                                                       view_height_px=10,
                                                       intrinsic=np.array([[5,0,5],
                                                                           [0,5,5],
                                                                           [0,0,1]]),
                                                       extrinsic=np.array([[1,0,0,0],
                                                                           [0,1,0,0],
                                                                           [0,0,1,0],
                                                                           [0,0,0,1]]),
                                                       scale=1.0)
# o3d.geometry.LineSet.create_camera_visualization

# o3d.visualization.draw_geometries([line_set, cam])


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False


o3d.visualization.draw_geometries_with_animation_callback([line_set, cam],
                                                          rotate_view)
