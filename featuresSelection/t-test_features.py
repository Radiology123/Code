import os
import pandas as pd
from scipy.stats import levene, ttest_ind

def process_file(file_path):
    # 创建T检验的输出子文件夹
    output_dir = "T检验"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    
    groups = df.groupby('Group')
    
    results = []  
    
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        try:
            group0_data = groups.get_group(0)[col].dropna()
            group1_data = groups.get_group(1)[col].dropna()
            
            if levene(group0_data, group1_data)[1] >= 0.05:
                t_stat, p_val = ttest_ind(group0_data, group1_data, equal_var=True)
            else:
                t_stat, p_val = ttest_ind(group0_data, group1_data, equal_var=False)
            
            mean_std_str_group0 = f"{group0_data.mean():.2f}±{group0_data.std():.2f}"
            mean_std_str_group1 = f"{group1_data.mean():.2f}±{group1_data.std():.2f}"
            results.append([col, mean_std_str_group0, mean_std_str_group1, t_stat, p_val])
            
        except Exception as e:
            print(f"Error processing column {col}: {e}")

    result_df = pd.DataFrame(results, columns=['Variable', 'Group0_Mean±Std', 'Group1_Mean±Std', 'T', 'P'])
    result_df.to_excel(os.path.join(output_dir, f'{file_name_without_ext}_tp_values.xlsx'), index=False)
    
    significant_cols = result_df[result_df['P'] < 0.05]['Variable'].tolist()
    merged_df = df[['Group', df.columns[1]] + significant_cols]
    merged_df.to_excel(os.path.join(output_dir, f'{file_name_without_ext}_经过t检验的特征矩阵.xlsx'), index=False)

# 使用os.walk遍历"data"子文件夹下的所有文件
for root, dirs, files in os.walk("data"):
    for file in files:
        if ('训练集' in file) and (file.endswith('.xlsx') or file.endswith('.csv')):
            file_path = os.path.join(root, file)
            process_file(file_path)