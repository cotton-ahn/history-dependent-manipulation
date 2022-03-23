import numpy as np

def generate_spatial_batch(N, featmap_H=256, featmap_W=256):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 2), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = (w / featmap_W) * 2 - 1
            xmax = ((w+1) / featmap_W) * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = (h / featmap_H) * 2 - 1
            ymax = ((h+1) / featmap_H) * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = [xmin, ymin]#, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val