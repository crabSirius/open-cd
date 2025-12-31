import os
import os.path as osp
import shutil
import cv2
import numpy as np
from typing import List, Tuple
from opencd.apis import OpenCDInferencer
from inference import get_all_checkpoints, get_input_size, get_config_from_checkpoint


def crop_image_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    input_size: int,
    overlap: int
) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    """
    将图像对裁剪成多个 patch，使用滑动窗口方式，重叠 overlap 像素
    
    Args:
        img1 (np.ndarray): 第一张图像
        img2 (np.ndarray): 第二张图像
        input_size (int): 裁剪尺寸
        overlap (int): 重叠像素数
    
    Returns:
        List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]: 
            patch 列表，每个元素为 (img1_patch, img2_patch, (y, x)) 坐标
    """
    h, w = img1.shape[:2]
    step = input_size - overlap
    patches = []
    
    # 计算所有需要裁剪的位置
    y_positions = []
    y = 0
    while y < h:
        y_positions.append(y)
        if y + input_size >= h:
            # 最后一个 patch，确保覆盖到图像末尾
            if y + input_size > h:
                y = max(0, h - input_size)
                if y not in y_positions:
                    y_positions.append(y)
            break
        y += step
    
    x_positions = []
    x = 0
    while x < w:
        x_positions.append(x)
        if x + input_size >= w:
            # 最后一个 patch，确保覆盖到图像末尾
            if x + input_size > w:
                x = max(0, w - input_size)
                if x not in x_positions:
                    x_positions.append(x)
            break
        x += step
    
    # 对每个位置进行裁剪
    for y in y_positions:
        for x in x_positions:
            y_end = min(y + input_size, h)
            x_end = min(x + input_size, w)
            
            # 裁剪 patch
            patch1 = img1[y:y_end, x:x_end]
            patch2 = img2[y:y_end, x:x_end]
            
            # 如果 patch 尺寸不足，进行填充
            if patch1.shape[0] < input_size or patch1.shape[1] < input_size:
                patch1_padded = np.zeros((input_size, input_size, 3), dtype=img1.dtype)
                patch2_padded = np.zeros((input_size, input_size, 3), dtype=img2.dtype)
                patch1_padded[:patch1.shape[0], :patch1.shape[1]] = patch1
                patch2_padded[:patch2.shape[0], :patch2.shape[1]] = patch2
                patch1 = patch1_padded
                patch2 = patch2_padded
            
            patches.append((patch1, patch2, (y, x)))
    
    return patches


def merge_patches(
    patches: List[np.ndarray],
    positions: List[Tuple[int, int]],
    original_shape: Tuple[int, int],
    input_size: int,
    overlap: int
) -> np.ndarray:
    """
    合并多个 patch 的结果，处理重叠区域
    
    Args:
        patches (List[np.ndarray]): patch 结果列表
        positions (List[Tuple[int, int]]): 每个 patch 的 (y, x) 坐标
        original_shape (Tuple[int, int]): 原始图像尺寸 (h, w)
        input_size (int): patch 尺寸
        overlap (int): 重叠像素数
    
    Returns:
        np.ndarray: 合并后的结果图像
    """
    h, w = original_shape
    # 使用 OR 逻辑：只要任何一个 patch 在某个位置是 255，合并后的结果就是 255
    merged = np.zeros((h, w), dtype=np.uint8)
    
    for patch, (y, x) in zip(patches, positions):
        # 确保 patch 是单通道
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # 计算实际可用的区域
        y_end = min(y + input_size, h)
        x_end = min(x + input_size, w)
        patch_h = y_end - y
        patch_w = x_end - x
        
        # 裁剪 patch 到实际大小
        patch_cropped = patch[:patch_h, :patch_w]
        
        # 使用 OR 逻辑：取最大值（只要有一个是 255，结果就是 255）
        merged[y:y_end, x:x_end] = np.maximum(merged[y:y_end, x:x_end], patch_cropped)
    
    return merged


def main():

    image_pair_list = [
        ['datas/1_1.png', 'datas/1_2.png'],
        ['datas/1_2.png', 'datas/1_1.png'],
        ['datas/2_1.png', 'datas/2_2.png'],
        ['datas/2_2.png', 'datas/2_1.png'],
        ['datas/3_1_resize.jpg', 'datas/3_2.jpg'],
        ['datas/3_2.jpg', 'datas/3_1_resize.jpg'],
    ]
    output_path = 'output_crop'
       
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
        
        # 计算重叠像素数
        overlap = input_size // 4
        
        # 创建临时目录用于存放 patch
        temp_dir = osp.join(output_path, 'temp_patches')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 处理每个图像对
        for image_pair in image_pair_list:
            # 读取原始图像
            img1 = cv2.imread(image_pair[0])
            img2 = cv2.imread(image_pair[1])
            
            if img1 is None or img2 is None:
                print(f"无法读取图像对: {image_pair}")
                continue
            
            original_shape = img1.shape[:2]  # (h, w)
            
            # 裁剪成多个 patch
            patches = crop_image_pair(img1, img2, input_size, overlap)
            print(f"图像对 {image_pair} 裁剪成 {len(patches)} 个 patch")
            
            # 保存 patch 到临时目录并准备推理
            patch_pair_list = []
            patch_positions = []
            for idx, (patch1, patch2, (y, x)) in enumerate(patches):
                patch1_path = osp.join(temp_dir, f'patch_{idx}_img1.jpg')
                patch2_path = osp.join(temp_dir, f'patch_{idx}_img2.jpg')
                cv2.imwrite(patch1_path, patch1)
                cv2.imwrite(patch2_path, patch2)
                patch_pair_list.append([patch1_path, patch2_path])
                patch_positions.append((y, x))
            
            # 对 patch 进行推理
            results = inferencer(image_pair_list, show=False, out_dir=output_path, return_datasamples=True)
            
            # 读取所有 patch 的推理结果
            patch_results = []
            for idx, (y, x) in enumerate(patch_positions):
                patch_result_path = osp.join(temp_dir, 'vis', f'patch_{idx}_img2.jpg')
                if osp.exists(patch_result_path):
                    # 读取 BGR 图像
                    patch_result = cv2.imread(patch_result_path)
                    if patch_result is not None:
                        # 使用 OR 方式：只要检测到任何一个类别（不是纯黑色），就设置为 255
                        # palette=[[0, 0, 0], [255, 255, 255]] 表示 unchanged=黑色, changed=白色
                        # 只要不是 [0, 0, 0]，就设置为 [255, 255, 255]
                        patch_result_or = np.zeros((patch_result.shape[0], patch_result.shape[1]), dtype=np.uint8)
                        # 检查每个像素，如果不是纯黑色，就设置为白色
                        non_black = np.any(patch_result != [0, 0, 0], axis=2) if len(patch_result.shape) == 3 else patch_result != 0
                        patch_result_or[non_black] = 255
                        patch_results.append(patch_result_or)
                    else:
                        # 如果读取失败，创建一个全零的 patch
                        patch_results.append(np.zeros((input_size, input_size), dtype=np.uint8))
                else:
                    patch_results.append(np.zeros((input_size, input_size), dtype=np.uint8))
            
            # 合并所有 patch 的结果
            merged_mask = merge_patches(patch_results, patch_positions, original_shape, input_size, overlap)
            
            # 保存合并后的结果
            output_sub_path = osp.join(output_path, osp.basename(config_path))
            os.makedirs(output_sub_path, exist_ok=True)
            os.makedirs(osp.join(output_sub_path, 'vis'), exist_ok=True)
            
            out_put_name = osp.basename(image_pair[1])
            merged_mask_path = osp.join(output_sub_path, 'vis', out_put_name)
            cv2.imwrite(merged_mask_path, merged_mask)
            
            # 生成可视化结果
            img2_res = img2.copy()
            mask_3ch = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR)
            img2_res[mask_3ch != 255] = (img2_res[mask_3ch != 255] * 0.3).astype(np.uint8)
            
            result = cv2.hconcat([img1, img2_res, img2])
            result = cv2.resize(result, (int(result.shape[1] * 0.5), int(result.shape[0] * 0.5)))
            
            result_name = osp.splitext(out_put_name)[0] + '_result.jpg'
            cv2.imwrite(osp.join(output_sub_path, result_name), result)
        
        # 清理临时目录
        # if osp.exists(temp_dir):
        #     shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
