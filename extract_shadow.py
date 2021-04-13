from misc.imutils import im2arr, save_image, get_connetcted_info, get_center_mask
from misc.pyutils import get_paths, mkdir
import ntpath
import os
import numpy as np
import cv2


def get_minRect(binary, out=None):
    """
    Obtain the smallest bounding rectangle of the irregular area
    input: binary map
    Returns: binary map (with the smallest bounding rectangle)
    """
    if out is None:
        out = np.zeros_like(binary,dtype=np.uint8)
    cnts,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)  # ret：中心坐标(x,y)，长宽（width, height）,总有 width>=height，角度angle
        box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        box=np.int0(box)
        cv2.fillPoly(out, [box], 1, lineType=8, shift=0) # 填充多边形区域某颜色
    return out

def get_overlap(mask1, mask2):
    """得到两个mask的重合部分，
    input：two binary maps, ndarray
    Returns：overlap of the two binary maps
    """
    ret = (mask1==mask2) * (mask1==1)
    return ret


def extract_shadows(image, mask, d=11):
    """
    Args:
        image: ndarray, uint8,  H*W*3
        mask: ndarray, uint8, H*W
        d:  Hyperparameters to increase the extension distance, depends on different datasets
             LEVIR-CD：d=11
             WHU-CD: d=41
    Returns: a label map with [building(1)+shadow(2)]
    """
    out = np.array(image < 50, dtype=np.uint8)
    out = out[:, :, 0] * out[:, :, 1] * out[:, :, 2]
    # There is no shadow inside the mask
    out[mask != 0] = 0
    # mask向外扩充N个像素，防止距离mask过远的阴影干扰
    kernel = np.ones((d, d), np.uint8)
    mask_extend = cv2.dilate(mask, kernel, iterations=1)
    mask_extend[mask == 1] = 0
    out[mask_extend == 0] = 0

    # Temporarily use for subsequent shadow judgment
    mask_extend_tmp = cv2.dilate(mask, np.ones((d//2, d//2), np.uint8), iterations=1)
    mask_extend_tmp[mask == 1] = 0

    _, _, stats_mask = get_connetcted_info(mask)
    x1, y1, dx, dy, area = stats_mask[1]
    mask_w, mask_h = dx, dy
    num_labels, labels, stats = get_connetcted_info(out)
    for i in range(1, num_labels):
        x1, y1, dx, dy, area = stats[i]
        x_center = x1 + dx // 2
        y_center = y1 + dy // 2
        shadow = np.array(labels == i, dtype=np.uint8)
        # Delete the connected component whose center is not in the mask
        # Generally, for building object, the center of the connected component is in the mask
        if (mask[y_center, x_center] == 0):
            out[labels == i] = 0
            # Rule out some manslaughter
            # 长度或宽度，与mask中心图形相近的，认为是阴影
            if (dx / mask_w > 0.4) or (dy / mask_h > 0.4):
                out[labels == i] = 1
        # 对于联通域中心在mask内的情况，需要排除一些虚警阴影，比如：远离建筑区域的阴影；
        else:
            # 对于建筑外沿区域，并且在候选阴影区域最小外接矩形中，包含大量非阴影区域的，需要排除
            shadow_rect = get_minRect(shadow)
            overlap = get_overlap(mask_extend_tmp, shadow_rect)
            number_nons = (out[overlap == 1] == 0).sum()
            number_rets = (overlap == 1).sum()
            print(number_nons, number_rets, number_nons / number_rets)
            if (number_nons / number_rets) > 0.3:
                out[labels == i] = 0
    # Complement the void hole inside the shadow
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Further Complement the hole
    # that may be caused by the inaccurate labeling of the edge of the building.
    # Now the edge of the building is enlarged
    out[mask != 0] = 1
    out_before = out.copy()
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # 定义矩形结构元素
    #     out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((1, 3), np.uint8)
    out = cv2.dilate(out, kernel, iterations=1)
    out = cv2.erode(out, kernel, iterations=1)
    out[(out - out_before) == 1] = 2

    out[mask != 0] = 2
    return out


if __name__ == '__main__':
    image_folder = r'samples\shadow_sample\image'
    label_folder = r'samples\shadow_sample\label'

    src_paths = get_paths(image_folder, '*.png')
    label_paths = get_paths(label_folder, '*.png')

    out_folder = r'samples\shadow_sample\shadow'
    mkdir(out_folder)

    assert src_paths.__len__() <= label_paths.__len__()
    for i, src_path in enumerate(src_paths):
        basename = ntpath.basename(src_path)
        print('process: ', src_path)
        src = im2arr(src_path)

        mask = im2arr(os.path.join(label_folder,basename))

        mask = get_center_mask(mask)
        out = extract_shadows(src, mask)

        save_image(out * 100, os.path.join(out_folder, basename))

