import os
import shutil
import re

# 源文件夹路径
source_folders = [
    r'./训练集预测概率',
    r'./验证集预测概率'
]

# 目标文件夹的基本路径
target_folder_base = r'./data'

# 创建一个正则表达式模式，用于匹配"modality"后跟一个或多个数字
pattern_modality = re.compile(r'modality\d+')

# 创建一个正则表达式模式，用于匹配"验证集"或"训练集"
pattern_dataset = re.compile(r'验证集|训练集')

# 遍历所有源文件夹
for source_folder in source_folders:
    # 遍历源文件夹中的文件
    for file in os.listdir(source_folder):
        # 使用正则表达式查找所有匹配的modality
        modalities_matches = pattern_modality.findall(file)
        dataset_match = pattern_dataset.search(file)

        if modalities_matches and dataset_match:
            modalities = "+".join(sorted(set(modalities_matches)))  # 删除重复的modality并排序
            dataset = dataset_match.group()
            
            new_filename = f"{modalities}_{dataset}.csv"
            target_folder = os.path.join(target_folder_base, modalities)

            # 如果目标文件夹不存在，则创建它
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            source_path = os.path.join(source_folder, file)
            target_path = os.path.join(target_folder, new_filename)
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            print(f'复制文件 {file} 到 {target_path}')

print('操作完成！')