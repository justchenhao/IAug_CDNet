from misc.pyutils import get_paths, mkdir, seed_random
from misc.imutils import im2arr, save_image

import ntpath
import os
import random
import numpy as np
import cv2

# Set random seed
seed_random(2020)

"""
This script is used to synthesize new CD data, i.e.,
add the existing target image/label pair to the existing CD data
"""


def blend(src, mask, dst, blend_mode='direct', expand_for_building=True):
    """
    :param src: ndarray h*w*3
    :param mask:ndarray h*w
    :param dst:ndarray h*w*3
    :return:  image blended by src and dst
    """
    assert src.shape[0] == mask.shape[0]
    assert src.shape[0] == dst.shape[0]
    assert src.shape[1] == mask.shape[1]
    assert src.shape[1] == dst.shape[1]
    assert mask.max() == 1
    dh, dw = src.shape[:2]

    # alpha_A
    mask_A = np.zeros([dh, dw],dtype=np.float)
    mask_A[mask == 1] = 1
    #  alpha_B
    mask_B = np.ones([dh, dw], dtype=np.float)
    mask_B[mask == 1] = 0

    if blend_mode is 'direct' or blend_mode=='poisson':
        expand_for_building = False

    if expand_for_building:
        kernel = np.ones((7, 7), np.uint8)
        mask_expand = cv2.dilate(mask, kernel, iterations=1)
        mask_expand_minus_ = mask_expand - mask
        #  中间过度地带
        mask_A[mask_expand_minus_ == 1] = 1
        mask_B[mask_expand_minus_ == 1] = 0

    if blend_mode == 'poisson':
        center = (dw//2, dh//2)
        mask_ = mask.copy() * 255
        expand = True
        if expand:
            # when levir-cd: d = 9
            # when whu-cd: d=15
            d = 9
            kernel = np.ones((d, d), np.uint8)
            mask_ = cv2.dilate(mask_, kernel, iterations=1)
        out = cv2.seamlessClone(src, dst, mask_, center, cv2.NORMAL_CLONE)
        return out

    elif blend_mode == 'gaussian':
        mask_A = cv2.GaussianBlur(mask_A, (7, 7), 2)
        mask_B = cv2.GaussianBlur(mask_B, (7, 7), 2)

    elif blend_mode == 'box':
        mask_A = cv2.blur(mask_A, (7, 7))
        mask_B = cv2.blur(mask_B, (7, 7))

    # extend to 3D
    mask_A = mask_A[:,:,np.newaxis]
    mask_B = mask_B[:, :, np.newaxis]
    mask_A = np.repeat(mask_A, repeats=3, axis=2)
    mask_B = np.repeat(mask_B, repeats=3, axis=2)

    out = src * mask_A + dst * mask_B
    return out.astype(np.uint8)


def sample_area(labels, dx, dy):
    """
    在h*w的labels中的背景区域中采样一个dx*dy大小的区域,
    最多尝试10次，如果采样失败，则返回None，None
    :param labels: ndarray，h*w，其中labels为0的区域被背景，其余为前景区域
    :return:(x,y)
    """
    h, w = labels.shape[:2]
    # random generate x, y
    y = random.randint(0, h-dy-1)
    x = random.randint(0, w-dx-1)
    #  Determine whether the area is in the labels area
    try_num = 0
    while (try_num < 10):
        if (labels[y:y + dy, x:x + dx].sum()==0):
            return x, y
        else:
            try_num += 1
            y = random.randint(0, h - dy - 1)
            x = random.randint(0, w - dx - 1)
    return None, None


def extend_bbox(x,y,dx,dy,h,w,extend_num):
    """
    extend area of the instance to size of extend_num*2
    :return:
    """
    out_x = x - extend_num
    if out_x < 0:
        out_x = 0
    out_y = y - extend_num
    if out_y < 0:
        out_y = 0
    out_dx = dx + extend_num*2
    if x+out_dx > w-1:
        out_dx = w - 1 - x
    out_dy = dy + extend_num * 2
    if y+out_dy > h-1:
        out_dy = h-1-y
    return out_x,out_y,out_dx,out_dy


def get_single_source_list(img_folder, label_folder, match_key='*.png', shuffle=True):
    """
    获取image和label文件夹中的文件列表，返回每对路径的list
    :param img_folder: str
    :param label_folder: str
    :param shuffle: bool
    :return:
    """
    img_paths = get_paths(img_folder, match_key)
    label_paths = get_paths(label_folder, match_key)
    assert img_paths.__len__() == label_paths.__len__()
    img_label_paths = list(zip(img_paths, label_paths))
    if shuffle:
        random.shuffle(img_label_paths)
    return img_label_paths


def generate_new_sample(out_A, out_B, out_L, A, B, ref, src, mask, blend_mode=None, mask_mode=None):
    """once paste one instance，
    ndarray: a reference value
    :return:
    """
    rotate_max_degree = 0
    if rotate_max_degree != 0:
        from misc.imutils import random_rotate_list
        [src], [mask] = random_rotate_list(
            ([src], [mask]), rotate_max_degree, (0, 0))
    scale = 1
    if scale != 1:
        from misc.imutils import random_scale_list
        [src], [mask] = random_scale_list(
            ([src], [mask]), [0.9, 1.1], (3, 0))

    mask_instance = mask
    if mask_mode == 'shadow':
        true_mask = np.array((mask == 200), np.uint8) * 255
    else:
        true_mask = (mask == 255).astype(np.uint8) * 255

    x1, y1, dx, dy = 0, 0, mask.shape[1], mask.shape[0]
    # sample x, y
    x, y = sample_area(ref, dx, dy)
    if x is None:
        return None
    #  random t1/t2
    if (random.random() > 0.5):
        out_A[y:y + dy, x:x + dx] = blend(src=src[y1:y1 + dy, x1:x1 + dx],
                                          mask=(mask_instance[y1:y1 + dy, x1:x1 + dx] != 0).astype(np.uint8),
                                          dst=A[y:y + dy, x:x + dx],
                                          expand_for_building=True,
                                          blend_mode=blend_mode)
    else:
        out_B[y:y + dy, x:x + dx] = blend(src=src[y1:y1 + dy, x1:x1 + dx],
                                          mask=(mask_instance[y1:y1 + dy, x1:x1 + dx] != 0).astype(np.uint8),
                                          dst=B[y:y + dy, x:x + dx],
                                          expand_for_building=True,
                                          blend_mode=blend_mode)
    out_L[y:y + dy, x:x + dx] = true_mask[y1:y1 + dy, x1:x1 + dx]
    ref[y:y + dy, x:x + dx] = true_mask[y1:y1 + dy, x1:x1 + dx]

    return 1


def syn_CD_data():
    ################################
    #  first define the some paths
    A_folder = r'samples\LEVIR\A'
    B_folder = r'samples\LEVIR\B'
    L_folder = r'samples\LEVIR\label'
    ref_folder = r'samples\LEVIR\ref'
    #  instance path
    src_folder = r'samples\SYN_CD\image' #test
    label_folder = r'samples\SYN_CD\shadow'  # test
    out_folder = r'samples\SYN_CD\out_sample'
    os.makedirs(out_folder, exist_ok=True)
    # how many instance to paste per sample
    M = 50
    ################################

    suffix = '*.png'
    image_read_mode = 3 if 'tif' in suffix else 1
    A_paths = get_paths(A_folder, suffix)
    B_paths = get_paths(B_folder, suffix)
    L_paths = get_paths(L_folder, suffix)
    ref_paths = get_paths(ref_folder, suffix)

    mask_mode = 'shadow'
    seed_random(2020)
    # load instances from folder
    src_label_paths = get_single_source_list(src_folder, label_folder,
                                             match_key='*.png', shuffle=True)

    modes = ['gaussian', 'poisson', 'extend']  # ['poisson','gaussian','extend','direct']

    for mode in modes:
        # valid pasted instance number
        n_valid = 0
        # Global counter of pasted instance
        n = 0

        seed_random(2020000000 + 26 + 3 + 12)

        out_path = os.path.join(out_folder, mode + '_' + mask_mode)

        out_path_A = os.path.join(out_path, 'A')
        out_path_B = os.path.join(out_path, 'B')
        out_path_L = os.path.join(out_path, 'label')
        mkdir(out_path_A)
        mkdir(out_path_B)
        mkdir(out_path_L)

        blend_method = mode

        output_txt = os.path.join(out_path, 'method_add_instance_from_' + blend_method + '_' + mask_mode + '_instancePerImage' + str(
                                      M) + '_log.txt')
        with open(output_txt, 'w'):
            pass

        assert A_paths.__len__() == L_paths.__len__()
        assert A_paths.__len__() == B_paths.__len__()
        assert A_paths.__len__() == ref_paths.__len__()

        for i, A_path in enumerate(A_paths):
            print('process: ', A_path)
            B_path = B_paths[i]
            L_path = L_paths[i]
            ref_path = ref_paths[i]

            basename = ntpath.basename(A_path)
            A = im2arr(A_path, mode=image_read_mode)
            B = im2arr(B_path, mode=image_read_mode)

            L = im2arr(L_path, mode=image_read_mode)
            ref = im2arr(ref_path, mode=image_read_mode).copy()

            out_A = A.copy()
            out_B = B.copy()
            out_L = L.copy()

            j = 0
            try_time = 0
            log_ins_list = []
            # attempt time on more than try_time_max
            while (j < M and try_time < 2000):
                src_path, label_path = src_label_paths[n % src_label_paths.__len__()]
                src = im2arr(src_path)
                mask = im2arr(label_path)
                n += 1
                if generate_new_sample(out_A, out_B, out_L, out_A, out_B, ref, src, mask, blend_mode=blend_method,
                                       mask_mode=mask_mode) is not None:
                    j += 1
                    log_ins_list.append(src_path + '\n')  # add log
                else:
                    try_time += 1

            # record copy-paste object info.
            log_add_instance = os.path.join(out_path, 'add_instances_log.txt')
            with open(log_add_instance, 'a') as log:
                for item in log_ins_list:
                    log.write(item)
                log.write('============================')
            # record pasted object numbers
            log_add_instance = os.path.join(out_path, 'add_instances_nums_log.txt')
            with open(log_add_instance, 'a') as log:
                item = len(log_ins_list)
                log.write(basename + ' ' + str(item) + '\n')
            n_valid = n_valid + j
            save_image(out_A, os.path.join(out_path_A, '' + basename))
            save_image(out_B, os.path.join(out_path_B, '' + basename))
            save_image(out_L, os.path.join(out_path_L, '' + basename))

        # record the number of the pasted object for the whole dataset
        log_add_instance = os.path.join(out_path, 'add_instances_nums_log.txt')
        with open(log_add_instance, 'a') as log:
            log.write('total paste instances num: %s' % n_valid)


if __name__ == '__main__':
    syn_CD_data()