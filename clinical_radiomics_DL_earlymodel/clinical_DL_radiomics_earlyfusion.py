import os
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from scipy.stats import ttest_ind, levene, pearsonr, spearmanr
from scipy import stats
from threading import Thread
import seaborn as sns
import matplotlib.pyplot as plt
# 相关性分析
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy import stats
import scipy
import os
# from skopt import BayesSearchCV
# from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef
from skopt.space import Real, Integer
from skopt import BayesSearchCV
from skopt.space import Categorical, Real
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import joblib
import scipy.stats as st
from sklearn.calibration import calibration_curve
# import random
import shap



def process_file(modality_num, filename):
    # 读取excel文件
    df = pd.read_excel(filename)

    # 指定不需要进行正则化的列
    exclude_cols = [0, 1]
    # 剩余为需要正则化的列
    index_dict[modality_num] = df.columns[2:]
    # # 批量正则化
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #
    # for i, col in enumerate(df.columns):
    #     if i in exclude_cols:
    #         continue
    #     # 使用MinMaxScaler进行缩放
    #     df[col] = scaler.fit_transform(df[[col]])


# step 2 直接读入训练集验证集
def preprocess_file(filename, column_indices):
    # 根据文件的扩展名选择读取方法
    if filename.endswith('.csv'):
        read_func = pd.read_csv
    elif filename.endswith('.xlsx'):
        read_func = pd.read_excel
    else:
        raise ValueError(f"Unsupported file format for file {filename}")

    # 使用选择的读取方法读取数据
    df = read_func(filename, usecols=[0])  # 1的话是imagename
    df_selected = read_func(filename, usecols=list(column_indices))

    # 将非数值型数据转换为NaN
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

    # 对选定的列进行标准化处理
    df_selected_scaled = preprocessing.scale(df_selected)
    df_selected_scaled = pd.DataFrame(df_selected_scaled, columns=df_selected.columns)

    # 将标准化处理后的数据与第一列拼接在一起
    result_df = pd.concat([df, df_selected_scaled], axis=1)

    return result_df


def process_modality(modality_number, column_indices, filename):
    # 为模态调用函数
    df_train = preprocess_file(filename[0], column_indices)
    df_val = preprocess_file(filename[1], column_indices)

    # 保存为Excel文件
    df_train.to_csv(f'Modality{modality_number}_Train_Features深+临.xlsx')
    df_val.to_csv(f'Modality{modality_number}_Validation_Features深+临.xlsx')

    return df_train, df_val


def process_all_modalities(index_dict, filenames):
    modalities_data = {}
    for modality_number, column_indices in index_dict.items():
        train_df, val_df = process_modality(modality_number, column_indices, filenames[modality_number - 1])
        modalities_data[modality_number] = {'train': train_df, 'val': val_df}
    return modalities_data


def get_X_y(train_df, val_df):
    X_train = train_df.drop('Group', axis=1)
    y_train = train_df['Group']

    X_val = val_df.drop('Group', axis=1)
    y_val = val_df['Group']

    return X_train, y_train, X_val, y_val


# 模型训练
def process_models(split_data, seed_range, csv_filename, model_name, model_info):
    X_train, y_train, X_test1, y_test1 = split_data
    print(f"Processing for {csv_filename}")
    # 准备保存结果的数据框
    param_results = pd.DataFrame()
    performance_results = pd.DataFrame()
    best_model_estimators = {}
    # 循环遍历不同的随机种子
    for seed in seed_range:

        print(f"Processing seed {seed}")
        # 随机种子设置
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # 在这里定义一个新的DataFrame，用来存储测试集的预测的概率
        test_proba_results = pd.DataFrame()

        # 在这里定义一个新的DataFrame，用来存储训练集的预测的概率
        train_proba_results = pd.DataFrame()
        # for model_name, model_info in models_params.items():

        print(f"Processing {model_name}")
        model = model_info['model']
        param_grid = model_info['params']
        bcv = BayesSearchCV(model, param_grid, cv=skf,
                            n_iter=30)  # you can change n_iter as per your requirement.
        bcv.fit(X_train, y_train)
        # 在这里将最佳模型实例保存到字典中
        best_model_estimators[model_name] = bcv.best_estimator_
        # Save the best parameters and the seed
        best_params = bcv.best_params_
        best_params['Seed'] = seed
        best_params['Model'] = model_name
        param_results = param_results.append(best_params, ignore_index=True)

        y_train_pred = cross_val_predict(bcv.best_estimator_, X_train, y_train, cv=5)
        y_test_pred = bcv.best_estimator_.predict(X_test1)

        y_train_scores = cross_val_predict(bcv.best_estimator_, X_train, y_train, cv=5, method='predict_proba')[
                         :, 1]
        y_test_scores = bcv.best_estimator_.predict_proba(X_test1)[:, 1]

        # 获取混淆矩阵
        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test1, y_test_pred)

        # 计算各个组件
        TP_train, FP_train, FN_train, TN_train = train_cm.ravel()
        TP_test, FP_test, FN_test, TN_test = test_cm.ravel()

        # 计算指标
        # PPV and NPV
        npv_train, ppv_train = TN_train / (FN_train + TN_train), TP_train / (TP_train + FP_train)
        npv_test, ppv_test = TN_test / (FN_test + TN_test), TP_test / (TP_test + FP_test)

        # Sensitivity (Recall) and Specificity
        sensitivity_train = TP_train / (TP_train + FN_train)
        specificity_train = TN_train / (TN_train + FP_train)
        sensitivity_test = TP_test / (TP_test + FN_test)
        specificity_test = TN_test / (TN_test + FP_test)

        # F1 score
        f1_train = 2 * (ppv_train * sensitivity_train) / (ppv_train + sensitivity_train)
        f1_test = 2 * (ppv_test * sensitivity_test) / (ppv_test + sensitivity_test)

        # Youden Index
        youden_train = sensitivity_train + specificity_train - 1
        youden_test = sensitivity_test + specificity_test - 1

        # MCC
        mcc_train = matthews_corrcoef(y_train, y_train_pred)
        mcc_test = matthews_corrcoef(y_test1, y_test_pred)

        # 计算置信区间
        confidence = 0.95
        train_auc = roc_auc_score(y_train, y_train_scores)
        test_auc = roc_auc_score(y_test1, y_test_scores)

        n_train = len(y_train_scores)
        m_train = train_auc
        std_err_train = stats.sem(y_train_scores)
        ci_train = std_err_train * stats.t.ppf((1 + confidence) / 2, n_train - 1)

        train_ci_lower = m_train - ci_train
        train_ci_upper = m_train + ci_train

        n_test = len(y_test_scores)
        m_test = test_auc
        std_err_test = stats.sem(y_test_scores)
        ci_test = std_err_test * stats.t.ppf((1 + confidence) / 2, n_test - 1)

        test_ci_lower = m_test - ci_test
        test_ci_upper = m_test + ci_test
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test1, y_test_pred)
        train_precision = precision_score(y_train, y_train_pred)
        test_precision = precision_score(y_test1, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test1, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test1, y_test_pred)

        # 在这里添加你的模型对测试集的预测概率
        y_test_proba = bcv.best_estimator_.predict_proba(X_test1)[:, 1]
        test_proba_results[model_name] = y_test_proba

        # 在这里添加你的模型对训练集的预测概率
        y_train_proba = bcv.best_estimator_.predict_proba(X_train)[:, 1]
        train_proba_results[model_name] = y_train_proba
        # 在这里添加你的模型对测试集的预测概率
        y_test_proba = bcv.best_estimator_.predict_proba(X_test1)
        test_proba_results[model_name + '_0'] = y_test_proba[:, 0]
        test_proba_results[model_name + '_1'] = y_test_proba[:, 1]

        # 在这里添加你的模型对训练集的预测概率
        y_train_proba = bcv.best_estimator_.predict_proba(X_train)
        train_proba_results[model_name + '_0'] = y_train_proba[:, 0]
        train_proba_results[model_name + '_1'] = y_train_proba[:, 1]
        # 整合真实y值、预测的y值和预测概率
        combined_test_results = pd.DataFrame({
            'Group': y_test1,
            'predict': y_test_pred,
            'pre_score': list(zip(y_test_proba[:, 0], y_test_proba[:, 1]))
        })
        # 整合训练集的真实y值、预测的y值和预测概率
        combined_train_results = pd.DataFrame({
            'Group': y_train,
            'predict': y_train_pred,
            'pre_score': list(zip(y_train_proba[:, 0], y_train_proba[:, 1]))
        })
        # 将所有结果添加到结果表中
        result_table = [model_name, train_accuracy, test_accuracy, train_precision, test_precision,
                        train_recall, test_recall, train_f1, test_f1, train_auc, test_auc, train_ci_lower,
                        train_ci_upper, test_ci_lower, test_ci_upper, npv_train, ppv_train, npv_test, ppv_test,
                        sensitivity_train, specificity_train, sensitivity_test, specificity_test, f1_train,
                        f1_test, youden_train, youden_test, mcc_train, mcc_test]
        current_results = pd.DataFrame([result_table],
                                       columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Train Precision',
                                                'Test Precision', 'Train Recall', 'Test Recall',
                                                'Train F1-score', 'Test F1-score', 'Train AUC', 'Test AUC',
                                                'Train AUC 95% CI Lower', 'Train AUC 95% CI Upper',
                                                'Test AUC 95% CI Lower', 'Test AUC 95% CI Upper', 'Train NPV',
                                                'Train PPV', 'Test NPV', 'Test PPV', 'Train Sensitivity',
                                                'Train Specificity', 'Test Sensitivity', 'Test Specificity',
                                                'Train F1 Score', 'Test F1 Score', 'Train Youden Index',
                                                'Test Youden Index', 'Train Matthews Correlation Coefficient',
                                                'Test Matthews Correlation Coefficient'])
        current_results['Seed'] = seed
        performance_results = pd.concat([performance_results, current_results], ignore_index=True)



        # 将模型参数、模型结果、训练集和测试集的预测概率保存到指定的CSV或Excel文件中
        param_results.to_csv(f'{csv_filename}_模型最佳参数.csv', index=False)
        performance_results.to_csv(f'{csv_filename}_不同随机种子下的模型结果.csv', index=False)
        train_proba_results.to_excel(f'{csv_filename}_训练集预测概率.xlsx', index=False)
        test_proba_results.to_excel(f'{csv_filename}_验证集预测概率.xlsx', index=False)
        # 保存每个模态下的真实y值和模型预测的y值
        y_train.to_csv(f'{csv_filename}_y_train.csv', index=False)
        pd.DataFrame(y_train_pred).to_csv(f'{csv_filename}_y_train_pred.csv', index=False)
        y_test1.to_csv(f'{csv_filename}_y_test.csv', index=False)
        pd.DataFrame(y_test_pred).to_csv(f'{csv_filename}_y_test_pred.csv', index=False)
        # 将训练集的整合结果保存到CSV文件中
        combined_train_results.to_csv(f'{csv_filename}_训练集的真实分组和模型预测的分组.csv', index=False)
        combined_test_results.to_csv(f'{csv_filename}_验证集的真实分组和模型预测的分组.csv', index=False)

        # 筛选条件
    # 找到Test AUC的上下界
    # 找到 Test AUC 的上下界
    lower_bound = performance_results['Train AUC'].mean() - 10
    upper_bound = performance_results['Train AUC'].mean() + 10

    # 选择 Test AUC 在这个范围内的模型，并且 Train AUC 也在这个范围内
    filtered_performance = performance_results[
        (performance_results['Train AUC'].between(lower_bound, upper_bound)) &
        (performance_results['Test AUC'].between(lower_bound, upper_bound))]

    # 如果存在符合条件的模型，从中选择Test AUC最高的那个
    if not filtered_performance.empty:
        best_auc_row = filtered_performance.loc[filtered_performance['Test AUC'].idxmax()]
    else:
        best_auc_row = performance_results.loc[performance_results['Test AUC'].idxmax()]

    # Print the seed and the corresponding performance
    best_seed = best_auc_row['Seed']
    print(f"The best seed is {best_seed}")

    best_performance = performance_results[performance_results['Seed'] == best_seed]
    print("The performance of each model with this seed is:")
    print(best_performance)

    # The model with the best performance
    best_model = best_performance.loc[best_performance['Test AUC'].idxmax(), 'Model']
    print(f"The model with the best performance is {best_model}")
    # 在循环结束后，你可以通过模型名称从字典中获取模型实例
    best_model_estimator = best_model_estimators[best_model]
    # 最后，你可以返回你需要的结果，例如最好的模型和相应的表现
    # 将 best_model_estimators 也作为一个返回值
    return best_model, best_performance, best_model_estimators


def plot_roc_curve(model, model_name, X, y, color):
    y_scores = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_scores)

    # 分步计算AUC
    auc_value_from_curve = np.trapz(tpr, fpr)

    # 直接使用roc_auc_score计算AUC
    auc_value_direct = roc_auc_score(y, y_scores)

    print(f"AUC from curve: {auc_value_from_curve}, AUC direct: {auc_value_direct}")

    plt.plot(fpr, tpr, label='{} (AUC = {:.3f})'.format(model_name, auc_value_direct), color=color, linewidth=2)


#def extract_data_and_plot_roc(filename, color, label):
    # 从文件中读取数据
  #  combined_results = pd.read_csv(filename)

  #  y_true = combined_results['Group']
  #  y_pred = combined_results['predict']
  #  y_score = combined_results['pre_score'].apply(lambda x: eval(x)[1])

   # fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
   # ACC = accuracy_score(y_true, y_pred)
   # AUC = auc(fpr, tpr)

   # plt.plot(fpr, tpr, color=color, label=f'{label} ACC={ACC:.3f} AUC={AUC:.3f}')


class DelongTest():
    def __init__(self, preds1, preds2, label, threshold=0.05):
        '''
        preds1:the output of model1
        preds2:the output of model2
        label :the actual label
        '''
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        # self._show_result()

    def _auc(self, X, Y) -> float:
        return 1 / (len(X) * len(Y)) * sum([self._kernel(x, y) for x in X for y in Y])

    def _kernel(self, X, Y) -> float:
        '''
        Mann-Whitney statistic
        '''
        return .5 if Y == X else int(Y < X)

    def _structural_components(self, X, Y) -> list:
        V10 = [1 / len(Y) * sum([self._kernel(x, y) for y in Y]) for x in X]
        V01 = [1 / len(X) * sum([self._kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B) -> float:
        return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])

    def _z_score(self, var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5) + 1e-8)

    def _group_preds_by_label(self, preds, actual) -> list:
        X = [p for (p, a) in zip(preds, actual) if a]
        Y = [p for (p, a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)

        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        # Compute entries of covariance matrix S (covar_AB = covar_BA)
        var_A = (self._get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_A01,
                                                                                                    auc_A,
                                                                                                    auc_A) * 1 / len(
            V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10) + self._get_S_entry(V_B01, V_B01,
                                                                                                    auc_B,
                                                                                                    auc_B) * 1 / len(
            V_B01))
        covar_AB = (self._get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10) + self._get_S_entry(V_A01, V_B01,
                                                                                                       auc_A,
                                                                                                       auc_B) * 1 / len(
            V_A01))

        # Two tailed test
        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z)) * 2

        return z, p


def proces_figure(X_tests, y_tests, data_type):
    # y_tests = [y_test1_1, y_test1_2, y_test1_3, y_test1_4]
    # X_tests = [X_test1_1, X_test1_2, X_test1_3, X_test1_4]
    for i in range(len(best_estimators)):
        for j in range(i + 1, len(best_estimators)):
            preds1 = best_estimators[i].predict_proba(X_tests[i])[:, 1]
            preds2 = best_estimators[j].predict_proba(X_tests[j])[:, 1]
            delong = DelongTest(preds1, preds2, y_tests[i])  # 使用模型i的y_test
            z, p = delong._compute_z_p()

    # 创建一个空的DataFrame来存储结果
    results = pd.DataFrame(columns=['modality序号', 'Modality序号', 'Z值', 'p-value'])

    for i in range(len(best_estimators)):
        for j in range(i + 1, len(best_estimators)):
            preds1 = best_estimators[i].predict_proba(X_tests[i])[:, 1]
            preds2 = best_estimators[j].predict_proba(X_tests[j])[:, 1]
            delong = DelongTest(preds1, preds2, y_tests[i])  # 使用模型i的y_test
            z, p = delong._compute_z_p()

            # 将结果添加到DataFrame中
            results = results.append({
                'modality序号': i + 1,
                'Modality序号': j + 1,
                'Z值': z,
                'p-value': p
            }, ignore_index=True)

    # 打印DataFrame
    results.to_csv(f'影+深+临验证集delong{data_type}.csv')
    print(results)


def calculate_net_benefit_model(thresh_group, model, X_test, y_test):
    y_scores = model.predict_proba(X_test)[:, 1]
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred = (y_scores > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total = len(y_test)
        net_benefit = (tp / total) - (fp / total) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_test):
    net_benefit_all = np.array([])
    tp = np.sum(y_test == 1)
    fp = np.sum(y_test == 0)
    total = len(y_test)
    for thresh in thresh_group:
        net_benefit = (tp / total) - (fp / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, model, X_test, y_test, model_name, model_names):
    colors = ['crimson', 'blue', 'green', 'purple', 'orange', 'black', 'dodgerblue']
    net_benefit_model = calculate_net_benefit_model(thresh_group, model, X_test, y_test)
    model_color = colors[model_names.index(model_name)]
    print(f"Model {model_name} is assigned the color {model_color}")  # add this line
    ax.plot(thresh_group, net_benefit_model, color=model_color, label=model_name)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.6)
    ax.set_xlabel('High Risk Threshold', fontdict={'family': 'Times New Roman', 'fontsize': 13})
    ax.set_ylabel('Net Benefit', fontdict={'family': 'Times New Roman', 'fontsize': 13})
    ax.grid('off')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc='upper right')

def process_dca(models, model_names, test_sets, set_type="validation"):
    # DCA曲线：验证集
    # 定义模型和模型名称列表
    # models = [best_estimators_1[best_model_1],
    #           best_estimators_2[best_model_2],
    #           best_estimators_3[best_model_3],
    #           best_estimators_4[best_model_4],
    #           best_estimators_5[best_model_5],
    #           best_estimators_6[best_model_6],
    #           best_estimators_7[best_model_7]]

    # models = [best_estimators_1[best_model_1]
    #           # best_estimators_2[best_model_2],
    #           # best_estimators_3[best_model_3],best_estimators_4[best_model_4]
    #           ]
    # model_names = ['Model Clinic', 'Model T1','Model T2','Model T1C','Model Clinic + T1','Model Clinic + T2','Model Clinic + T1C']
    # model_names = ['Model Clinic', 'Model Clinic+Rad', 'Model Clinic+DL', 'Model Clinic+Rad+DL']
    # 定义测试数据
    # test_sets = [(X_test1_1, y_test1_1),
    #              (X_test1_2, y_test1_2),
    #              (X_test1_3, y_test1_3),
    #              (X_test1_4, y_test1_4),
    #             (X_test1_5, y_test1_5),
    #              (X_test1_6, y_test1_6),
    #              (X_test1_7, y_test1_7)]
    # test_sets = [(X_test1_1, y_test1_1)
    #              # (X_test1_2, y_test1_2),
    #              # (X_test1_3, y_test1_3),(X_test1_4, y_test1_4)
    #              ]
    # 定义阈值组
    thresh_group = np.arange(0, 1, 0.01)

    # 开始绘制
    fig, ax = plt.subplots(figsize=(10, 8))

    for model, model_name, test_set in zip(models, model_names, zip(*test_sets)):
        X_test, y_test = test_set
        plot_DCA(ax, thresh_group, model, X_test, y_test, model_name, model_names)  # change here

    # 在循环结束后绘制"All True"和"None"线
    net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
    ax.plot(thresh_group, net_benefit_all, color='black', linestyle='--', label='All True')
    ax.plot((0, 1), (0, 0), color='grey', linestyle=':', label='None')

    # 添加一些额外的样式和标题
    ax.grid(False)
    ax.set_title('DCA Curve for Models', fontsize=16)
    ax.legend(loc="upper right", fontsize=12)

# 保存DCA曲线图像
    filename = f"{set_type}_DCA_Models.pdf"
    path = f"dca_curve/{set_type}/"
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, format="pdf")
    print(f"DCA curve has been saved to {full_path}.")

def calculate_std_err(y_test, y_score, prob_pred):
    # 获取bin的边界
    bin_edges = np.unique([0, *prob_pred, 1])

    # 计算属于每个bin的样本的索引
    bin_indices = np.digitize(y_score, bin_edges[1:-1])

    # 初始化列表存储每个bin的标准误差
    std_errs = []

    for bin_index in range(len(prob_pred)):
        # 获取当前bin的实际值
        actual_values = y_test[bin_indices == bin_index]

        # 只有当 bin 中有值时才计算标准误差
        if len(actual_values) > 0:
            std_err = np.std(actual_values) / np.sqrt(len(actual_values))
            std_errs.append(std_err)
        else:
            std_errs.append(0)

    return std_errs


def predict_models(models, test_sets):
    # 初始化一个列表来存储每个模型的预测结果
    y_scores = []

    # 对于每个模型，使用它在测试集上进行预测，并将预测概率添加到列表中
    for model, test_set in zip(models, zip(*test_sets)):
        X_test, _ = test_set
        y_score = model.predict_proba(X_test)[:, 1]
        y_scores.append(y_score)

    return y_scores


# models = [best_estimators_1[best_model_1],
#           best_estimators_2[best_model_2],
#           best_estimators_3[best_model_3],
#           best_estimators_4[best_model_4],
#           best_estimators_5[best_model_5],
#           best_estimators_6[best_model_6],
#           best_estimators_7[best_model_7]]
# model_names = ['Model Clinic', 'Model T1','Model T2','Model T1C','Model Clinic + T1','Model Clinic + T2','Model Clinic + T1C']

# # 定义测试数据
# test_sets = [(X_test1_1, y_test1_1),
#              (X_test1_2, y_test1_2),
#              (X_test1_3, y_test1_3),
#              (X_test1_4, y_test1_4),
#             (X_test1_5, y_test1_5),
#              (X_test1_6, y_test1_6),
#              (X_test1_7, y_test1_7)]
def process_align(models, model_names, test_sets, set_type="validation"):
    # models = [best_estimators_1[best_model_1],
    #           best_estimators_2[best_model_2],
    #           best_estimators_3[best_model_3], best_estimators_4[best_model_4]
    #           ]
    # model_names = ['Model Clinic', 'Model T1','Model T2','Model T1C','Model Clinic + T1','Model Clinic + T2','Model Clinic + T1C']
    # model_names = ['Model Clinic', 'Model Clinic+Rad', 'Model Clinic+DL', 'Model Clinic+Rad+DL']
    # 定义测试数据
    # test_sets = [(X_test1_1, y_test1_1),
    #              (X_test1_2, y_test1_2),
    #              (X_test1_3, y_test1_3),
    #              (X_test1_4, y_test1_4),
    #             (X_test1_5, y_test1_5),
    #              (X_test1_6, y_test1_6),
    #              (X_test1_7, y_test1_7)]
    # test_sets = [(X_test1_1, y_test1_1),
    #              (X_test1_2, y_test1_2),
    #              (X_test1_3, y_test1_3), (X_test1_4, y_test1_4)
    #              ]
    # 获取模型预测结果
    y_scores = predict_models(models, test_sets)

    # 初始化图像
    plt.figure(figsize=(8, 8))

    # 对每个模型进行绘图
    for index, test_set in enumerate(zip(*test_sets)):
        _, y_test = test_set
        prob_true, prob_pred = calibration_curve(y_test, y_scores[index], n_bins=5)
        std_errs = calculate_std_err(y_test, y_scores[index], prob_pred)
        scaled_std_errs = [err * 0.5 for err in std_errs]  # 乘以0.5作为示例，你可以选择其他缩放因子
        # 使用调整后的参数绘制误差线
        plt.errorbar(prob_pred, prob_true, yerr=scaled_std_errs, fmt='o', color=colors[index], label=model_names[index],
                     elinewidth=1, capsize=3)
        plt.plot(prob_pred, prob_true, color=colors[index])  # 添加这一行来连接点

    # 绘制理想的校准曲线
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

    # 添加图例和标签
    plt.legend(loc="upper left")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
# 保存图像为 SVG
    filename_svg = f"{set_type}_align_curve.svg"
    filename_pdf = f"{set_type}_align_curve.pdf"
    path = f"align_curve/{set_type}/"
    if not os.path.exists(path):
        os.makedirs(path)

    full_path_svg = os.path.join(path, filename_svg)
    full_path_pdf = os.path.join(path, filename_pdf)

    plt.savefig(full_path_svg, format='svg')
    plt.savefig(full_path_pdf, format='pdf')
    print(f"Align curve has been saved to {full_path_svg} and {full_path_pdf}.")
    #plt.show()


def process_confusion(X_tests, y_tests, modal_names, data_type):
    # model_names = [best_model_1, best_model_2, best_model_3, best_model_4]  # 模型名称列表
    group_names = ['0', '1']  # 分组名称列表
    # X_tests = [X_test1_1, X_test1_2, X_test1_3, X_test1_4]  # 测试集列表
    # y_tests = [y_test1_1, y_test1_2, y_test1_3, y_test1_4]  # 测试集标签列表

    # 对每个模态进行循环，绘制混淆矩阵
    # for i, model in enumerate([best_estimators_1[best_model_1], best_estimators_2[best_model_2],
    #                            best_estimators_3[best_model_3], best_estimators_4[best_model_4], best_estimators_5[best_model_5], best_estimators_6[best_model_6], best_estimators_7[best_model_7]]):
    for i, model in enumerate(best_estimators):
        y_pred = model.predict(X_tests[i])
        cm = confusion_matrix(y_tests[i], y_pred)
        plt.figure(figsize=(5, 4))
        # plt.title(f'Model: {model_names[i]}')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                plt.text(k, j, cm[j, k], ha="center", va="center", color="white" if cm[j, k] > thresh else "black")
        plt.xticks(np.arange(len(group_names)), group_names, rotation=45)
        plt.yticks(np.arange(len(group_names)), group_names)
        plt.xlabel('Predicted group')
        plt.ylabel('True group')
        plt.title(f'Confusion Matrix for Modality{i + 1}')

        plt.savefig(f'./{modal_names[i]}{data_type}-混淆矩阵-Modality{i + 1}-{modal_names[i]}.svg', format='svg', dpi=1200,
                    bbox_inches='tight')
        #plt.show()


# 模型解释器
def explain_model_with_shap(best_model_names, best_model_estimators, X_tests, X_trains):
    shap.initjs()

    for i, (model_name, model) in enumerate(zip(best_model_names, best_model_estimators)):
        print(f"Processing SHAP for Modality {i + 1} with model {model_name}")

        if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
        elif model_name in ["SVM", "KNN", "Logistic"]:
            explainer = shap.KernelExplainer(model.predict_proba, X_trains[i])
        else:
            explainer = shap.KernelExplainer(model.predict, X_trains[i])

        shap_values = explainer.shap_values(X_tests[i])

        # SHAP Summary Plot for class 1
        plt.figure()
        tmp_shap_value_0 = shap_values[1] if isinstance(shap_values,list) else shap_values
        shap.summary_plot(tmp_shap_value_0, X_tests[i], show=False)
        # shap.summary_plot(shap_values[1], X_tests[i], show=False)
        plt.savefig(f'./SHAP-Summary-Class1-Modality{i + 1}-{model_name}.svg', format='svg', dpi=1200,
                    bbox_inches='tight')
       # plt.show()

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_tests[i], show=False)
        plt.savefig(f'./SHAP-Summary-Modality{i + 1}-{model_name}.svg', format='svg', dpi=1200, bbox_inches='tight')
        #plt.show()

        # For binary classification, particularly with SVM
        if model_name == "SVM" or isinstance(shap_values,
                                             list):  ##############在这里填模型名称，和上面保持一致！和第9、11行一致！##############
            plt.figure()
            shap.summary_plot(shap_values[0], X_tests[i], show=False)
            plt.savefig(f'./SHAP-Summary-0-Modality{i + 1}-{model_name}.svg', format='svg', dpi=1200,
                        bbox_inches='tight')
            #plt.show()

        # Dependency plot for each feature
        for feature in X_tests[i].columns:
            plt.figure()
            tmp_shap_value_1 = shap_values[0] if isinstance(shap_values,list) else shap_values
            shap.dependence_plot(feature, tmp_shap_value_1, X_tests[i], show=False)
            plt.savefig(f'./SHAP-Dependency-{feature}-Modality{i + 1}-{model_name}.svg', format='svg', dpi=1200,
                        bbox_inches='tight')
            #plt.show()

        # SHAP Force Plot for a specific sample
        #         sample_index = 12  # You can customize this index
        #         shap.plots.force(explainer.expected_value[1], shap_values[1][sample_index, :], X_tests[i].iloc[sample_index, :], matplotlib=True)

        # SHAP Decision Plots for 10 random samples
        sample_indices = list(range(19, 30))  ##################### 从第20到第30个样本，可以自己定######################
        for sample in sample_indices:
            plt.figure()
            tmp_shap_value_2 = shap_values[1] if isinstance(shap_values, list) else shap_values
            tmp_expect_value_3 = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            shap.decision_plot(tmp_expect_value_3, tmp_shap_value_2[sample, :], X_tests[i].columns,
                               link='logit', show=False)
            plt.savefig(f'./SHAP-Decision-Sample{sample}-Modality{i + 1}-{model_name}.svg', format='svg', dpi=1200,
                        bbox_inches='tight')
            #plt.show()



if __name__ == "__main__":
    filenames = []
    if len(filenames) == 0:
        datapath = './data'
        modal_names = []
        for modal in os.listdir(datapath):
            modal_names.append(modal)
            tmp = []
            for file in os.listdir(os.path.join(datapath, modal)):
                tmp.append(os.path.join(datapath, modal, file))
            if '训练' in tmp[0] or 'train' in tmp[0]: 
                filenames.append(tmp)
            else: 
                filenames.append(tmp[::-1])
    
    index_dict = {}
    for i, train_val in enumerate(filenames):
        # 根据文件扩展名选择读取方法
        if train_val[0].endswith('.csv'):
            df = pd.read_csv(train_val[0])
        elif train_val[0].endswith('.xlsx'):
            df = pd.read_excel(train_val[0])
        else:
            raise ValueError(f"Unsupported file format for file {train_val[0]}")
        
        index_dict[i + 1] = df.columns[2:]

    for i in range(len(filenames)):
        print(f"For modality {i + 1}, features: ")
        print(index_dict[i + 1])

    # step 2
    # 使用从Lasso回归模型中动态提取的特征列的索引
    modalities_data = process_all_modalities(index_dict, filenames)

    # 现在你可以通过模态的编号来访问对应的训练集和验证集
    # modality_1_train_df, modality_1_val_df = modalities_data[1]['train'], modalities_data[1]['val']
    # modality_2_train_df, modality_2_val_df = modalities_data[2]['train'], modalities_data[2]['val']

    # 划分每个模态的训练集验证集
    split_datas = []
    for modal, train_val in modalities_data.items():
        split_datas.append(get_X_y(train_val['train'], train_val['val']))  # X_train, y_train, X_test, y_test


    # 定义随机种子范围
    seed_range = range(200, 210)
    # 需要根据自己的模态改, 对模态1,2……进行process_models
    modality_rets = []
    # Prepare the models and their parameter grids
    models_params = {
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'C': [0.1, 200],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': Integer(1, 200),
                'max_depth': Integer(1, 200),
                'min_samples_split': Integer(2, 200),
            }
        },
        'SGD': {
            'model': SGDClassifier(),
            'params': {
                'loss': Categorical(['log']),
                'penalty': Categorical(['l1', 'l2', 'elasticnet']),
                'alpha': Real(0.0001, 1),
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': Integer(1, 10),
                'weights': Categorical(['uniform', 'distance']),
                'algorithm': Categorical(['auto', 'kd_tree', 'brute'])
            }
        },
        'XGBoost': {
            'model': XGBClassifier(),
            'params': {
                'n_estimators': Integer(1, 200),
                'max_depth': Integer(1, 200),
                'learning_rate': Real(0.0001, 1),
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(),
            'params': {
                'n_estimators': Integer(1, 200),
                'max_depth': Integer(1, 200),
                'learning_rate': Real(0.0001, 1),
            }
        }
    }
    all_selected = {0: 'SVM', 1: 'RandomForest', 2: 'SGD', 3: 'KNN', 4: 'XGBoost', 5: 'LightGBM'}
    while True:
        select_model_num = int(
            input('[0] SVM\n[1] RandomForest\n[2] SGD\n[3] KNN\n[4] XGBoost\n[5] LightGBM\n 请输入要选择的模型前的序号(输入0-5中的一个数字)'))
        if select_model_num in range(0, 6):
            break
        else:
            print('输入不合法，请重新输入')
            continue
    for idx, split_data in enumerate(split_datas):
        modality_rets.append(process_models(split_data, seed_range, f'modality{idx + 1}', all_selected[select_model_num], models_params[all_selected[select_model_num]]))
    # # 对模态1的数据处理
    # best_model_1, best_performance_1, best_estimators_1 = process_models(X_train_1, y_train_1, X_test1_1, y_test1_1,
    #                                                                      seed_range, 'modality1')
    #
    # # 对模态2的数据处理
    # best_model_2, best_performance_2, best_estimators_2 = process_models(X_train_2, y_train_2, X_test1_2, y_test1_2,
    #                                                                      seed_range, 'modality2')
    # 对模态3的数据处理
    # best_model_3, best_performance_3, best_estimators_3 = process_models(X_train_3, y_train_3, X_test1_3, y_test1_3, seed_range, 'modality3')

    # best_model_4, best_performance_4, best_estimators_4 = process_models(X_train_4, y_train_4, X_test1_4, y_test1_4, seed_range, 'modality4')
    # best_model_5, best_performance_5, best_estimators_5 = process_models(X_train_5, y_train_5, X_test1_5, y_test1_5, seed_range, 'modality5')
    # best_model_6, best_performance_6, best_estimators_6 = process_models(X_train_6, y_train_6, X_test1_6, y_test1_6, seed_range, 'modality6')
    # best_model_7, best_performance_7, best_estimators_7 = process_models(X_train_7, y_train_7, X_test1_7, y_test1_7, seed_range, 'modality7')
    # 定义颜色
    colors = ['#F0C808', '#00A6D6', '#8BC34A', '#FF5722', '#B71C1C', '#7B1FA2', 'dodgerblue']

    ##################### 验证集##################
    plt.figure(figsize=(8, 6))
    # 需要根据自己的模态改
    # 绘制模态1,2……的ROC曲线
    for idx, data in enumerate(split_datas):
        plot_roc_curve(modality_rets[idx][2][modality_rets[idx][0]], modal_names[idx], data[2], data[3], colors[idx])

    # # 绘制第一种模态的ROC曲线
    # plot_roc_curve(best_estimators_1[best_model_1], f"Model RAD", X_test1_1, y_test1_1, colors[0])
    #
    # # 绘制第二种模态的ROC曲线
    # plot_roc_curve(best_estimators_2[best_model_2], f"Model Clinic+Rad", X_test1_2, y_test1_2, colors[1])

    # plot_roc_curve(best_estimators_3[best_model_3], f"Model Clinic+DL", X_test1_3, y_test1_3, colors[2])
    # plot_roc_curve(best_estimators_4[best_model_4], f"Model Clinic+Rad+DL", X_test1_4, y_test1_4, colors[3])
    # plot_roc_curve(best_estimators_5[best_model_5], f"Model Clinic + T1 Rad", X_test1_5, y_test1_5, colors[4])
    # plot_roc_curve(best_estimators_6[best_model_6], f"Model Clinic + T2 Rad", X_test1_6, y_test1_6, colors[5])
    # plot_roc_curve(best_estimators_7[best_model_7], f"Model Clinic + T1C Rad", X_test1_7, y_test1_7, colors[6])

    fpr_range = np.linspace(0, 1, num=200)
    plt.plot(fpr_range, fpr_range, color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    # plt.title('ROC Curve for Models', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(False)

    # 保存ROC曲线图像
    filename = 'ROC-验证集-Models.pdf'
    path = "roc_curve/"
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, format="pdf")
    print(f"ROC curve has been saved to {full_path}.")
    #plt.show()

    ############################训练集##########################
    # 在训练集上绘制ROC曲线
    plt.figure(figsize=(8, 6))
    for idx, data in enumerate(split_datas):
        plot_roc_curve(modality_rets[idx][2][modality_rets[idx][0]], modal_names[idx], data[0], data[1], colors[idx])
    # plot_roc_curve(best_estimators_1[best_model_1], f"Model RAD", X_train_1, y_train_1, colors[0])
    #
    # # 绘制第二种模态的ROC曲线
    # plot_roc_curve(best_estimators_2[best_model_2], f"Model Clinic+Rad", X_train_2, y_train_2, colors[1])
    # #
    # plot_roc_curve(best_estimators_3[best_model_3], f"Model Clinic+DL",  X_train_3, y_train_3, colors[2])

    # plot_roc_curve(best_estimators_4[best_model_4], f"Model T1+T2+T3", X_train_4, y_train_4, colors[3], best_performance_4['Train AUC'].values[0])
    # plot_roc_curve(best_estimators_4[best_model_4], f"Model Clinic+Rad+DL",  X_train_4, y_train_4, colors[3])
    # plot_roc_curve(best_estimators_5[best_model_5], f"Model Clinic + T1 Rad",  X_train_5, y_train_5, colors[4])
    # plot_roc_curve(best_estimators_6[best_model_6], f"Model Clinic + T2 Rad",  X_train_6, y_train_6, colors[5])
    # plot_roc_curve(best_estimators_7[best_model_7], f"Model Clinic + T1C Rad",  X_train_7, y_train_7, colors[6])

    fpr_range = np.linspace(0, 1, num=200)
    plt.plot(fpr_range, fpr_range, color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    # plt.title('ROC Curve for Models', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(False)

    # 保存训练集ROC曲线图像
    filename = 'ROC-Models-训练.pdf'
    path = "roc_curve/"
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, format="pdf")
    print(f"Train ROC curve has been saved to {full_path}.")
    #plt.show()

    # cell 14
    # colors = ['#F0C808', '#00A6D6', '#8BC34A', '#FF5722', '#B71C1C', '#7B1FA2', 'dodgerblue']
    # 绘图
    plt.figure(figsize=(8, 6))

    # 提供模态的文件名、颜色和标签来绘制每个模态的ROC曲线
    # filenames = ['modality1_验证集_combined_results.csv', 'modality2_验证集_combined_results.csv']
    # labels = ["Model RAD", "Model Clinic+Rad"]
    #filenames_val = [f'modality{i + 1}_验证集_combined_results.csv' for i in range(len(modal_names))]
    #for i in range(len(filenames_val)):
    #    extract_data_and_plot_roc(filenames_val[i], colors[i], modal_names[i])

   # fpr_range = np.linspace(0, 1, num=200)
    #plt.plot(fpr_range, fpr_range, color='navy', linestyle='--')
   # plt.xlim([-0.05, 1.05])
   # plt.ylim([-0.05, 1.05])
   # plt.xlabel('False Positive Rate', fontsize=14)
   # plt.ylabel('True Positive Rate', fontsize=14)
   # plt.legend(loc="lower right", fontsize=12)
   # plt.grid(False)

    # 保存ROC曲线图像
    #filename = 'ROC-test-Models.pdf'
    #path = "roc_curve/"
   # if not os.path.exists(path):
   #     os.makedirs(path)
   # full_path = os.path.join(path, filename)
   # plt.savefig(full_path, format="pdf")
   # print(f"ROC curve has been saved to {full_path}.")
    #plt.show()
    # 保存模型
    for idx, modality_ret in enumerate(modality_rets):
        joblib.dump(modality_ret[2][modality_ret[0]], f'best_model_modality{idx + 1}.pkl')

    # delong检验：验证集


    # 将模型和测试集组织为列表
    best_estimators = [(modality_ret[2][modality_ret[0]]) for modality_ret in modality_rets]
    # best_model_1, best_performance_1, best_estimators_1
    # best_estimators = [best_estimators_1[best_model_1], best_estimators_2[best_model_2],
    #                    best_estimators_3[best_model_3], best_estimators_4[best_model_4]]
    # , best_estimators_4[best_model_4], best_estimators_5[best_model_5], best_estimators_6[best_model_6], best_estimators_7[best_model_7]
    # y_tests = [y_test1_1, y_test1_2, y_test1_3, y_test1_4, y_test1_5, y_test1_6, y_test1_7]
    # X_tests = [X_test1_1, X_test1_2, X_test1_3, X_test1_4, X_test1_5, X_test1_6, X_test1_7]


    # 验证集画图
    X_tests, y_tests = list(zip(*[(data[2], data[3]) for data in split_datas]))
    proces_figure(X_tests, y_tests, '验证')
    # 训练集画图
    X_trains, y_trains = list(zip(*[(data[0], data[1]) for data in split_datas]))
    proces_figure(X_trains, y_trains, '训练')


# DCA曲线：验证集
    test_sets = list(zip(*[(data[2], data[3]) for data in split_datas]))
    process_dca(best_estimators, modal_names, test_sets, "validation")

# DCA曲线：训练集
    train_sets = list(zip(*[(data[0], data[1]) for data in split_datas]))
    process_dca(best_estimators, modal_names, train_sets, "training")


    #########################################



# 校准曲线：验证集
    test_sets = list(zip(*[(data[2], data[3]) for data in split_datas]))
    process_align(best_estimators, modal_names, test_sets, "validation")

# 校准曲线：训练集
    train_sets = list(zip(*[(data[0], data[1]) for data in split_datas]))
    process_align(best_estimators, modal_names, train_sets, "training")
    # 混淆矩阵：验证集

    # X_tests, y_tests = list(zip(*[(data[2], data[3]) for data in split_datas]))
    process_confusion(X_tests, y_tests, modal_names, '验证集')
    # 混淆矩阵：训练集
    # X_trains, y_trains = list(zip(*[(data[0], data[1]) for data in split_datas]))
    process_confusion(X_trains, y_trains, modal_names, '训练集')
    explain_model_with_shap([all_selected[select_model_num]] * len(best_estimators), best_estimators, X_tests, X_trains)


