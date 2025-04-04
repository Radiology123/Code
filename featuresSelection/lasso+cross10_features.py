import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path
# 生成模拟数据

# 判断和创建目录
output_dir = "lasso回归特征筛选"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 获取"data"子文件夹下所有带有“训练集”字符的.xlsx和.csv文件
all_files = []
for root, dirs, files in os.walk("data"):
    for file in files:
        if ("训练集" in file) and (file.endswith('.xlsx') or file.endswith('.csv')):
            all_files.append(os.path.join(root, file))
for file_path in all_files:
    # Load data
    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

    # Extract data from dataframe
    X = df.iloc[:, 2:].values 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = df['Group']  # 预测 'Group'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 设置lambda值
    alphas = np.linspace(0, 0.5, 200)

    # 使用Lasso回归并进行交叉验证
    lasso = LassoCV(alphas=alphas, cv=10, random_state=0)
    lasso.fit(X_train, y_train)

    # 画图显示MSE随alpha的变化
    # 1. 为MSE曲线添加95%置信区间误差棒
    mse_mean = lasso.mse_path_.mean(axis=-1)
    mse_std = lasso.mse_path_.std(axis=-1)
    plt.figure(figsize=(6, 6))
    plt.plot(np.log1p(lasso.alphas_), mse_mean, label="Average MSE across folds")
    plt.fill_between(np.log1p(lasso.alphas_), mse_mean - 1.96 * mse_std, mse_mean + 1.96 * mse_std, color='gray', alpha=0.2)
    plt.axvline(np.log1p(lasso.alpha_), linestyle="--", color="red", label="Chosen alpha")
    plt.xlabel("log(Alpha)")
    plt.ylabel("Mean Square Error")
    plt.title("MSE vs. log(Alpha) with 95% CI")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name_without_ext}_cv_with_CI.pdf"))
    


    alphas_lasso, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=alphas)
    m_log_alphas = np.log1p(alphas_lasso)
    plt.figure(figsize=(6, 6))
    plt.plot(m_log_alphas, coefs_lasso.T)
    plt.axvline(np.log1p(lasso.alpha_), linestyle="--", color="red", label="Chosen alpha")
    plt.ylabel('Coefficients')
    plt.xlabel('log(Alpha)')
    plt.title('Lasso Coefficients Progression for Various Alphas')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name_without_ext}_lam.pdf"))
    #plt.show()
    # 保存系数如何随着lam的变化而变化
    # 保存系数如何随着lam的变化而变化
    # 2. 输出系数到CSV文件，并只选择非零系数并绘制图像
    # 获取特征名称
    feature_names = df.columns[2:]

    # 选取非零系数及其对应的特征名称
    non_zero_indices = np.where(lasso.coef_ != 0)[0]
    non_zero_coefs = lasso.coef_[non_zero_indices]
    non_zero_feature_names = feature_names[non_zero_indices]

    # 将特征名称和其对应的系数放到DataFrame中并输出到CSV文件
    coeff_df = pd.DataFrame({
        'Feature Name': non_zero_feature_names,
        'Coefficient': non_zero_coefs
    })
    coeff_df.to_csv(os.path.join(output_dir, f"{file_name_without_ext}_Lasso-回归筛选特征系数.csv"), index=False)

    # 绘制非零系数的图像
    sorted_indices = np.argsort(non_zero_coefs)
    sorted_coefs = non_zero_coefs[sorted_indices]
    x_values = np.arange(len(sorted_coefs))
    plt.figure(figsize=(6, 6))
    plt.bar(x_values, sorted_coefs, color=np.where(sorted_coefs > 0, 'firebrick', 'lightblue'), edgecolor='black', alpha=0.8)
    plt.xticks(x_values, sorted_indices, rotation=45, ha='right', va='top')
    plt.ylabel('weight')
    plt.savefig(os.path.join(output_dir, f"{file_name_without_ext}_Lasso-回归筛选特征系数柱状图.pdf"))
    #plt.show()

    # 3. 对于非零特征，输出相应的数据列和文件的第1和2列
    non_zero_indices = np.where(lasso.coef_ != 0)[0]
    selected_data = df.iloc[:, [0, 1] + list(non_zero_indices + 2)]  # 加2是因为我们跳过了前两列
    selected_data.to_csv(os.path.join(output_dir, f"{file_name_without_ext}_Lasso-特征矩阵.csv"), index=False)