from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model
import numpy as np
from os import path

dirname = path.dirname(__file__)

CONFIG_FILE = path.join(dirname, '..', 'models', 'segmentation', 'config.py')
CHECKPOINT_FILE = path.join(dirname, '..', 'models', 'segmentation', 'iter_160000.pth')

def get_pred_mask(img:np.ndarray) -> np.ndarray:

    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')
    model = revert_sync_batchnorm(model)

    result = inference_model(model, img)

    pred_mask = result.pred_sem_seg.data.numpy()
    # pred_mask = result.pred_sem_seg.data.cpu().numpy()
    
    return  np.squeeze(pred_mask, axis=0)