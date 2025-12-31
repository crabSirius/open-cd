import os
import os.path as osp
import shutil
import cv2
import numpy as np
from opencd.apis import OpenCDInferencer

from inference import get_all_checkpoints, get_input_size, get_config_from_checkpoint

def resize_image_pair(image_pair: list, input_size: int, temp_dir: str) -> list:
    """
    将图像对 resize 到 input_size * input_size 大小，并保存到临时目录
    
    Args:
        image_pair (list): 图像对路径列表 [img1_path, img2_path]
        input_size (int): 目标尺寸大小
        temp_dir (str): 临时目录路径
    
    Returns:
        list: resize 后的图像对路径列表
    """
    os.makedirs(temp_dir, exist_ok=True)
    resized_pair = []
    
    for img_path in image_pair:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # Resize 到 input_size * input_size
        img_resized = cv2.resize(img, (input_size, input_size))
        
        # 保存到临时目录
        img_name = osp.basename(img_path)
        temp_path = osp.join(temp_dir, img_name)
        cv2.imwrite(temp_path, img_resized)
        resized_pair.append(temp_path)
    
    return resized_pair

def main():

    image_pair_list = [
        ['datas/1_1.png', 'datas/1_2.png'],
        ['datas/1_2.png', 'datas/1_1.png'],
        ['datas/2_1.png', 'datas/2_2.png'],
        ['datas/2_2.png', 'datas/2_1.png'],
    ]
    output_path = 'output_resize'
       
    checkpoints = get_all_checkpoints()
    for checkpoint in checkpoints:
        # input_size = get_input_size(checkpoint)
        input_size = 1024
        config_path = get_config_from_checkpoint(checkpoint)
        if config_path:
            print(f"Checkpoint: {checkpoint}, Input Size: {input_size}, Config: {config_path}")
        else:
            print(f"Checkpoint: {checkpoint}, Input Size: {input_size}, Config: Not found")
            continue
        try:
            inferencer = OpenCDInferencer(model=config_path, weights=checkpoint, classes=('unchanged', 'changed'), palette=[[0, 0, 0], [255, 255, 255]])
        except Exception as e:
            print(f"Checkpoint: {checkpoint}, Input Size: {input_size}, Config: {config_path}, Error: {e}")
            continue
        
        # 创建临时目录用于存放 resize 后的图像
        temp_dir = osp.join(output_path, 'temp_resized_images')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 将图像对 resize 到 input_size * input_size
        resized_image_pair_list = []
        for image_pair in image_pair_list:
            resized_pair = resize_image_pair(image_pair, input_size, temp_dir)
            resized_image_pair_list.append(resized_pair)
        
        # 使用 resize 后的图像进行推理
        inferencer(resized_image_pair_list, show=False, out_dir=output_path)
        
        # 清理临时目录
        if osp.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # 将ouput_path 移动 到 output_path 的子目录中
        output_sub_path = osp.join(output_path, osp.basename(config_path))
        os.makedirs(output_sub_path, exist_ok=True)
        shutil.move(output_path + '/vis', output_sub_path)

        for image_pair in image_pair_list:
            out_put_name = osp.basename(image_pair[1])
            img1 = cv2.imread(image_pair[0])
            img2 = cv2.imread(image_pair[1])
            mask = cv2.imread(osp.join(output_sub_path + '/vis', out_put_name))
            
            # 使用 OR 方式：只要检测到任何一个类别（不是纯黑色），就设置为 255
            # palette=[[0, 0, 0], [255, 255, 255]] 表示 unchanged=黑色, changed=白色
            # 只要不是 [0, 0, 0]，就设置为 [255, 255, 255]
            if mask is not None:
                # 检查每个像素，如果不是纯黑色，就设置为白色
                mask_or = np.zeros_like(mask)
                # 对于 BGR 图像，检查是否不是 [0, 0, 0]
                if len(mask.shape) == 3:
                    non_black = np.any(mask != [0, 0, 0], axis=2)
                    mask_or[non_black] = [255, 255, 255]
                else:
                    # 灰度图
                    non_black = mask != 0
                    mask_or[non_black] = 255
                mask = mask_or
            
            mask_resize = cv2.resize(mask, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_NEAREST)

            img2_res = img2.copy()
            mask_3ch = mask_resize if len(mask_resize.shape) == 3 else cv2.cvtColor(mask_resize, cv2.COLOR_GRAY2BGR)
            img2_res[mask_3ch != 255] = (img2_res[mask_3ch != 255] * 0.3).astype(np.uint8)

            result = cv2.hconcat([img1, img2_res, img2])
            result = cv2.resize(result, (int(result.shape[1] * 0.5), int(result.shape[0] * 0.5)))

            result_name = osp.splitext(out_put_name)[0] + '_result.jpg'
            cv2.imwrite(osp.join(output_sub_path, result_name), result)

if __name__ == '__main__':
    main()
