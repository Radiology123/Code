import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, normaltest
from sklearn.cluster import KMeans
output_dir = "相关性筛选"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def process_selected_file(file_path):
    # Load data
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

    # Extract data from dataframe
    X = df.iloc[:, 2:].values 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check for normality
    is_normal = True
    significant_columns = df.columns[2:]
    for col in significant_columns:
        if not normaltest(df[col])[1] >= 0.05:
            is_normal = False
            break

    # Calculate correlation matrix
    corr_func = pearsonr if is_normal else spearmanr
    corr_values = []
    for col1 in significant_columns:
        corr_row = []
        for col2 in significant_columns:
            corr_val, _ = corr_func(df[col1], df[col2])
            corr_row.append(corr_val)
        corr_values.append(corr_row)
    corr_df = pd.DataFrame(corr_values, columns=significant_columns, index=significant_columns)

    # Reorder columns using K-means clustering
    kmeans = KMeans(n_clusters=3)
    y = kmeans.fit_predict(corr_df.values.T)
    new_order = y.argsort()
    corr_df = corr_df.iloc[new_order, :].iloc[:, new_order]

    # Identify and drop highly correlated features
    correlated_features = set()
    for i in range(len(corr_df.columns)):
        for j in range(i):
            if abs(corr_df.iloc[i, j]) > 0.9:
                colname = corr_df.columns[i]
                correlated_features.add(colname)

    selected_cols = [col for col in significant_columns if col not in correlated_features]
    selected_data = df[['Group', 'imageName'] + selected_cols]
    
    # 输出文件的命名考虑到了原始文件的名称
    output_filename = f"特征矩阵_{os.path.basename(file_path)}"
    output_filepath = os.path.join("相关性筛选", output_filename)
    selected_data.to_excel(output_filepath, index=False)

    # Plotting correlation heatmap of the selected columns
    plt.figure(figsize=(12, 12))
    sns.heatmap(selected_data.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap of Selected Features ({file_path})')
    plt.yticks(rotation=45)  # 旋转y轴的标签
    plt.xticks(rotation=45)  # 旋转x轴的标签
    plt.savefig(os.path.join("相关性筛选", f"heatmap_{os.path.basename(file_path).split('.')[0]}.png"))
    plt.savefig(os.path.join("相关性筛选", f"heatmap_{os.path.basename(file_path).split('.')[0]}.pdf"), dpi=500)
    plt.close()  # 关闭图形以释放内存

    print(f"Finished processing: {file_path}")
# 确保"data"目录存在
data_dir = "data"
if not os.path.exists(data_dir):
    print("Error: 'data' directory does not exist in the current path.")
else:
    # 使用os.walk遍历"data"目录及其所有子目录
    for root, _, files in os.walk(data_dir):
        for file in files:
            if ('训练集' in file) and (file.endswith('.xlsx') or file.endswith('.csv')):
                file_path = os.path.join(root, file)
                process_selected_file(file_path)