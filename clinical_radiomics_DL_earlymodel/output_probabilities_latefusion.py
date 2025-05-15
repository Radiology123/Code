def process_file(modality_num, filename):
    df = pd.read_excel(filename)

    exclude_cols = [0, 1]
    index_dict[modality_num] = df.columns[2:]
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #
    # for i, col in enumerate(df.columns):
    #     if i in exclude_cols:
    #         continue
    #     df[col] = scaler.fit_transform(df[[col]])


def preprocess_file(filename, column_indices):
    if filename.endswith('.csv'):
        read_func = pd.read_csv
    elif filename.endswith('.xlsx'):
        read_func = pd.read_excel
    else:
        raise ValueError(f"Unsupported file format for file {filename}")


    df = read_func(filename, usecols=[0])  # 1的话是imagename
    df_selected = read_func(filename, usecols=list(column_indices))
    df_selected = df_selected.apply(pd.to_numeric, errors='coerce')
    df_selected_scaled = preprocessing.scale(df_selected)
    df_selected_scaled = pd.DataFrame(df_selected_scaled, columns=df_selected.columns)


    result_df = pd.concat([df, df_selected_scaled], axis=1)

    return result_df


def process_modality(modality_number, column_indices, filename):

    df_train = preprocess_file(filename[0], column_indices)
    df_val = preprocess_file(filename[1], column_indices)


    df_train.to_csv(f'Modality{modality_number}_Train_Featuresd+c.xlsx')
    df_val.to_csv(f'Modality{modality_number}_Validation_Featuresd+c.xlsx')

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


def process_models(split_data, seed_range, csv_filename, model_name, model_info):
    X_train, y_train, X_test1, y_test1 = split_data
    print(f"Processing for {csv_filename}")

    param_results = pd.DataFrame()
    performance_results = pd.DataFrame()
    best_model_estimators = {}

    for seed in seed_range:

        print(f"Processing seed {seed}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

  
        test_proba_results = pd.DataFrame()

  
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


        train_cm = confusion_matrix(y_train, y_train_pred)
        test_cm = confusion_matrix(y_test1, y_test_pred)


        TP_train, FP_train, FN_train, TN_train = train_cm.ravel()
        TP_test, FP_test, FN_test, TN_test = test_cm.ravel()


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


        y_test_proba = bcv.best_estimator_.predict_proba(X_test1)[:, 1]
        test_proba_results[model_name] = y_test_proba


        y_train_proba = bcv.best_estimator_.predict_proba(X_train)[:, 1]
        train_proba_results[model_name] = y_train_proba

        y_test_proba = bcv.best_estimator_.predict_proba(X_test1)
        test_proba_results[model_name + '_0'] = y_test_proba[:, 0]
        test_proba_results[model_name + '_1'] = y_test_proba[:, 1]


        y_train_proba = bcv.best_estimator_.predict_proba(X_train)
        train_proba_results[model_name + '_0'] = y_train_proba[:, 0]
        train_proba_results[model_name + '_1'] = y_train_proba[:, 1]
        # 整合真实y值、预测的y值和预测概率
        combined_test_results = pd.DataFrame({
            'Group': y_test1,
            'predict': y_test_pred,
            'pre_score': list(zip(y_test_proba[:, 0], y_test_proba[:, 1]))
        })

        combined_train_results = pd.DataFrame({
            'Group': y_train,
            'predict': y_train_pred,
            'pre_score': list(zip(y_train_proba[:, 0], y_train_proba[:, 1]))
        })

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




        param_results.to_csv(f'{csv_filename}_1.csv', index=False)
        performance_results.to_csv(f'{csv_filename}_2.csv', index=False)
        train_proba_results.to_excel(f'{csv_filename}_3.xlsx', index=False)
        test_proba_results.to_excel(f'{csv_filename}_4.xlsx', index=False)

        y_train.to_csv(f'{csv_filename}_y_train.csv', index=False)
        pd.DataFrame(y_train_pred).to_csv(f'{csv_filename}_y_train_pred.csv', index=False)
        y_test1.to_csv(f'{csv_filename}_y_test.csv', index=False)
        pd.DataFrame(y_test_pred).to_csv(f'{csv_filename}_y_test_pred.csv', index=False)

        combined_train_results.to_csv(f'{csv_filename}_1.csv', index=False)
        combined_test_results.to_csv(f'{csv_filename}_2.csv', index=False)


    lower_bound = performance_results['Train AUC'].mean() - 10
    upper_bound = performance_results['Train AUC'].mean() + 10


    filtered_performance = performance_results[
        (performance_results['Train AUC'].between(lower_bound, upper_bound)) &
        (performance_results['Test AUC'].between(lower_bound, upper_bound))]


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

    best_model_estimator = best_model_estimators[best_model]

    return best_model, best_performance, best_model_estimators


def plot_roc_curve(model, model_name, X, y, color):
    y_scores = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_scores)


    auc_value_from_curve = np.trapz(tpr, fpr)


    auc_value_direct = roc_auc_score(y, y_scores)

    print(f"AUC from curve: {auc_value_from_curve}, AUC direct: {auc_value_direct}")

    plt.plot(fpr, tpr, label='{} (AUC = {:.3f})'.format(model_name, auc_value_direct), color=color, linewidth=2)


#def extract_data_and_plot_roc(filename, color, label):

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


    results = pd.DataFrame(columns=['modality序号', 'Modality序号', 'Z值', 'p-value'])

    for i in range(len(best_estimators)):
        for j in range(i + 1, len(best_estimators)):
            preds1 = best_estimators[i].predict_proba(X_tests[i])[:, 1]
            preds2 = best_estimators[j].predict_proba(X_tests[j])[:, 1]
            delong = DelongTest(preds1, preds2, y_tests[i])  # 使用模型i的y_test
            z, p = delong._compute_z_p()

            results = results.append({
                'modality序号': i + 1,
                'Modality序号': j + 1,
                'Z值': z,
                'p-value': p
            }, ignore_index=True)

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

    thresh_group = np.arange(0, 1, 0.01)


    fig, ax = plt.subplots(figsize=(10, 8))

    for model, model_name, test_set in zip(models, model_names, zip(*test_sets)):
        X_test, y_test = test_set
        plot_DCA(ax, thresh_group, model, X_test, y_test, model_name, model_names)  # change here


    net_benefit_all = calculate_net_benefit_all(thresh_group, y_test)
    ax.plot(thresh_group, net_benefit_all, color='black', linestyle='--', label='All True')
    ax.plot((0, 1), (0, 0), color='grey', linestyle=':', label='None')


    ax.grid(False)
    ax.set_title('DCA Curve for Models', fontsize=16)
    ax.legend(loc="upper right", fontsize=12)


    filename = f"{set_type}_DCA_Models.pdf"
    path = f"dca_curve/{set_type}/"
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    plt.savefig(full_path, format="pdf")
    print(f"DCA curve has been saved to {full_path}.")

def calculate_std_err(y_test, y_score, prob_pred):

    bin_edges = np.unique([0, *prob_pred, 1])


    bin_indices = np.digitize(y_score, bin_edges[1:-1])


    std_errs = []

    for bin_index in range(len(prob_pred)):

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

    y_scores = predict_models(models, test_sets)


    plt.figure(figsize=(8, 8))


    for index, test_set in enumerate(zip(*test_sets)):
        _, y_test = test_set
        prob_true, prob_pred = calibration_curve(y_test, y_scores[index], n_bins=5)
        std_errs = calculate_std_err(y_test, y_scores[index], prob_pred)
        scaled_std_errs = [err * 0.5 for err in std_errs]

        plt.errorbar(prob_pred, prob_true, yerr=scaled_std_errs, fmt='o', color=colors[index], label=model_names[index],
                     elinewidth=1, capsize=3)
        plt.plot(prob_pred, prob_true, color=colors[index]) 


    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')


    plt.legend(loc="upper left")
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
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
    # model_names = [best_model_1, best_model_2, best_model_3, best_model_4]  
    group_names = ['0', '1']  # 分组名称列表
    # X_tests = [X_test1_1, X_test1_2, X_test1_3, X_test1_4]  
    # y_tests = [y_test1_1, y_test1_2, y_test1_3, y_test1_4] 
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


