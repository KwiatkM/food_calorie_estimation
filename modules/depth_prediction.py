import torch
import cv2
import numpy as np
import open3d as o3d

print("CUDA available: "+ str(torch.cuda.is_available()))
print("CUDA device count: " + str(torch.cuda.device_count()))
print("CUDA current device: " + torch.cuda.get_device_name(torch.cuda.current_device()))

DETECTED_COIN_REAL_RADIUS = 1.2 # wartość w centymetrach

def predict(rgb_image, fx,fy,cx,cy)-> np.ndarray:

    h, w = rgb_image.shape[:2]
    intrinsic = [fx,fy,cx,cy]
    input_size = (616, 1064) # wybrane dla używanego modelu
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    padding = [123.675, 116.28, 103.53] # kolor
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # normalizacja
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    # predykcja
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model.cuda().eval()

    pred_depth, confidence, output_dict = model.inference({'input': rgb})

    # usunięcie paddingu
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]

    # przekształcenie do oryginalnego rozmiaru
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_image.shape[:2], mode='bilinear').squeeze()

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    return pred_depth.cpu().numpy()


def GeneratePointCloudsFromMask(input_image, depth_image, mask_image,fx,fy,cx,cy,coin=(0,0,0)):
    '''
    input_image - numpy HxWx3
    depth_image - numpy HxW
    mask_image - numpy HxW
    fx,fy - (ogniskowa kamery dla osi x/y) - float
    cx,cy - (punkt główny kamery - współrzędne x/y) - float
    coin - współrzędne wykrytej monety - tuple(y:int, x:int, r:int)
    '''
    width, height = input_image.shape[0:2]

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    unique_mask_values, pixel_counts = np.unique(mask_image, return_counts=True)
    n_of_pixels = width*height

    # skalowanie na podstawie znalezionej monety
    if not coin == (0,0,0):

        circle_roi = input_image[coin[1]-coin[2]:coin[1]+coin[2], coin[0]-coin[2]:coin[0]+coin[2]]
        depth_circle_roi = depth_image[coin[1]-coin[2]:coin[1]+coin[2], coin[0]-coin[2]:coin[0]+coin[2]]


        # cv2.imshow("Result", circle_roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        point_1_depth = depth_image[coin[1], coin[0]] 
        point_2_depth = depth_image[coin[1]+ coin[2], coin[0]]

        point_1_world = np.array([  (coin[1] - cx) * point_1_depth / fx,
                                    (coin[0] - cy) * point_1_depth / fy,
                                    point_1_depth])
        
        point_2_world = np.array([  (coin[1] + coin[2] - cx) * point_2_depth / fx,
                                    (coin[0] - cy) * point_2_depth / fy,
                                    point_2_depth])


        world_distance = np.linalg.norm(point_2_world - point_1_world)

        scale_factor = DETECTED_COIN_REAL_RADIUS / world_distance # wartość w centymetrach
        print(f"Znaleziono monetę. Współczynnik skali = {scale_factor}")

    else:
        print("Nie znaleziono monety. Obliczone wartości mogą być znacznie niedokładne")
        scale_factor = 1.0

    computed_mask_values = []
    point_clouds = []
    for index, value in enumerate(unique_mask_values):
        if value == 0: continue

        if pixel_counts[index]/n_of_pixels < 0.01:
            continue

        mask2D = (mask_image == value)
        segmented_image = input_image * np.dstack((mask2D ,mask2D ,mask2D))
        segmented_depth_map = depth_image * mask2D 
        
        depth_3d = o3d.geometry.Image(segmented_depth_map)
        image_3d = o3d.geometry.Image(segmented_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_3d, depth_3d, convert_rgb_to_intensity=False, depth_scale=1)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

        pcd.scale(scale_factor, pcd.get_center())
        point_clouds.append(pcd)
        computed_mask_values.append(value)

    
    return point_clouds, computed_mask_values


