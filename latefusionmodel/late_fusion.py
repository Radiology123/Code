import os
import shutil

import os
import shutil

base_path = r"./"

# 创建输出文件夹，如果它们不存在
train_folder = os.path.join(base_path, "训练集预测概率")
valid_folder = os.path.join(base_path, "验证集预测概率")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)

# 遍历所有子文件和子文件夹
for dirpath, dirnames, filenames in os.walk(base_path):
    for file in filenames:
        if file.endswith('.xlsx') or file.endswith('.csv'):
            source_path = os.path.join(dirpath, file)
            
            # 检查文件名并决定要移动到哪个文件夹
            if "训练集预测概率" in file:
                dest_path = os.path.join(train_folder, file)
            elif "验证集预测概率" in file:
                dest_path = os.path.join(valid_folder, file)
            else:
                continue

            # 如果目标文件夹中存在同名文件，跳过
            if not os.path.exists(dest_path):
                shutil.move(source_path, dest_path)
            else:
                print(f"跳过 {file}，因为它已经存在于目标文件夹中。")

print("移动文件成功-接下来进行后融合!")

import os
import pandas as pd
from itertools import combinations

# ① 列出所有文件
folder_path = r"./训练集预测概率"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.xlsx')]

# 为每个文件组合生成合并的文件
for r in range(2, len(all_files)+1):
    for subset in combinations(all_files, r):
        first_file = subset[0]
        df_combined = pd.read_excel(os.path.join(folder_path, first_file)) if first_file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, first_file))
        
        # ② ③ 按照给定的规则合并文件
        for file in subset[1:]:
            df = pd.read_excel(os.path.join(folder_path, file)) if file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, file))
            df_combined = pd.concat([df_combined, df.iloc[:, 2:]], axis=1)
        
        # ④ 保存合并后的文件
        output_name = "+".join([os.path.splitext(f)[0] for f in subset]) + '.csv'
        df_combined.to_csv(os.path.join(folder_path, output_name), index=False)

print("Files combined successfully!")



folder_path = r"./验证集预测概率"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.xlsx')]

# 为每个文件组合生成合并的文件
for r in range(2, len(all_files)+1):
    for subset in combinations(all_files, r):
        first_file = subset[0]
        df_combined = pd.read_excel(os.path.join(folder_path, first_file)) if first_file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, first_file))
        
        # ② ③ 按照给定的规则合并文件
        for file in subset[1:]:
            df = pd.read_excel(os.path.join(folder_path, file)) if file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, file))
            df_combined = pd.concat([df_combined, df.iloc[:, 2:]], axis=1)
        
        # ④ 保存合并后的文件
        output_name = "+".join([os.path.splitext(f)[0] for f in subset]) + '.csv'
        df_combined.to_csv(os.path.join(folder_path, output_name), index=False)

print("Files combined successfully!")


