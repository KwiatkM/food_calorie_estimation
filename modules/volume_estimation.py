import depth_prediction as dp
import coin_finder as cf
import open3d as o3d
import numpy as np
import copy
import pymeshfix
import cv2
import matplotlib.pyplot as plt

def predict_volume(image, mask,fx,fy,cx,cy, visualize:bool=False):
    h, w = image.shape[0:2]
    coin = cf.find_coin(image)

    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    depth = dp.predict(image,fx,fy,cx,cy)

    if visualize:
        pass
    # cv2.imshow("Result", depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pcds, meal_ids = dp.GeneratePointCloudsFromMask(image,depth,mask,fx,fy,cx,cy,coin)

    print(f"Rozpoznano: {meal_ids}")
   
    volumes = []
    for idx, pcd in enumerate(pcds):
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        # outlier removal
        cl, index = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=4.0)
        pcd = pcd.select_by_index(index)

        bb = pcd.get_oriented_bounding_box()
        bb.color = [0,0,1]

        # rotate to align with global coordintate system
        rotation_matrix = bb.R
        rotation_center = bb.center
        inverse_rotation_matrix = np.linalg.inv(bb.R)
        pcd.rotate(inverse_rotation_matrix, center=rotation_center)
        # bb.rotate(inverse_rotation_matrix, center=rotation_center)

        points = np.asarray(pcd.points)
        points_z = points[:,2]
        
        # usunięcie punktów o znacznie innej wartości Z
        z_mean = np.mean(points_z)
        z_std = np.std(points_z)
        z_min_threshold = z_mean - 3 * z_std
        z_max_threshold = z_mean + 3 * z_std
        points = points[(points_z >= z_min_threshold) & (points_z <= z_max_threshold)]
        pcd.points = o3d.utility.Vector3dVector(points)
        
        bb = pcd.get_axis_aligned_bounding_box()
        z_min = np.asarray(bb.get_box_points())[:,2].min()
        z_max = np.asarray(bb.get_box_points())[:,2].max()
        z_half = ((z_max - z_min) / 2) + z_min
        # print(f"z_min={z_min}, z_max={z_max}, z_half={z_half}")
        
        # sprawdzenie z której strony bb jest więcej punktów
        close_point_counter = 0
        far_point_counter = 0

        for p in points:
            if p[2] <= z_half:
                close_point_counter += 1
            else:
                far_point_counter += 1
        # print(f"close: {close_point_counter}, far: {far_point_counter}")
        if far_point_counter < close_point_counter:
            pcd.rotate([[1,  0,  0], [0, -1,  0], [0,  0, -1]], center=rotation_center)
            bb = pcd.get_axis_aligned_bounding_box()

        bb.color = [0.0,0.0,1.0]
        if visualize: o3d.visualization.draw_geometries([pcd, bb]) # ,mesh_frame


        points = np.asarray(pcd.points)
        z_min = np.asarray(bb.get_box_points())[:,2].min()
        z_max = np.asarray(bb.get_box_points())[:,2].max()
        # point projection on z plane
        points = copy.deepcopy(points)
        points[:,2] = z_min - (0.05 * (z_max-z_min)) # TODO: do zmiany
        # points[:,2] = z_max*1.01 # TODO: do zmiany
        pcd.points.extend(points)
        # pcd.colors.extend(pcd.colors)

        # calculate normals
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()

        # flip normals on base plane TODO: check if true for other images
        size = int(np.asarray(pcd.normals).shape[0] / 2)
        np.asarray(pcd.normals)[size:] *= -1

        bb.color = [0.0,0.0,1.0]
        if visualize: o3d.visualization.draw_geometries([pcd, bb], point_show_normal=True)

        # calculate mesh TODO: check why depth>5 = not wotertight
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5, n_threads=1)

        mesh.compute_vertex_normals()

        # remove disconnected parts of the mesh
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 1000 # TODO: check if 1000 is ok for other / smaller meshes
        mesh.remove_triangles_by_mask(triangles_to_remove)

        # o3d.visualization.draw_geometries([mesh])
        # rotate to original position
        
        # mesh.rotate(rotation_matrix, center=rotation_center)
        

        if mesh.is_watertight():
            volumes.append(mesh.get_volume())
            print(f"Obliczono objętość dla id={meal_ids[idx]}")

            if visualize: 
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
                o3d.visualization.draw_geometries([mesh,bb])
        else:
            # close mesh if not watertight
            mesh_v = np.asarray(mesh.vertices)
            mesh_f = np.asarray(mesh.triangles)
            meshfix = pymeshfix.MeshFix(mesh_v, mesh_f)
            meshfix.repair()

            fixed_mesh = o3d.geometry.TriangleMesh()
            fixed_mesh.vertices = o3d.utility.Vector3dVector(meshfix.v)
            fixed_mesh.triangles = o3d.utility.Vector3iVector(meshfix.f)

            fixed_mesh.compute_vertex_normals()
            fixed_mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
            if visualize: o3d.visualization.draw_geometries([fixed_mesh,bb])

            if fixed_mesh.is_watertight():
                volumes.append(fixed_mesh.get_volume())
                print(f"Obliczono objętość dla id={meal_ids[idx]}")
            else:
                volumes.append(0.0)
                print(f"Nie udało się obliczyć objętości dla id={meal_ids[idx]}")
        
        

    # if visualize:
    #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    #     visulaization_elements = []
    #     visulaization_elements.extend(meshes)
    #     o3d.visualization.draw_geometries(visulaization_elements, mesh_show_back_face=True)
    
    return volumes, meal_ids