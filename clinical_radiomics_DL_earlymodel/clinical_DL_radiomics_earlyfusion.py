
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

 
    df_train.to_csv(f'Modality{modality_number}_Train_FeaturesD+C.xlsx')
    df_val.to_csv(f'Modality{modality_number}_Validation_FeaturesD+C.xlsx')

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



        param_results.to_csv(f'{csv_filename}_模型最佳参数.csv', index=False)
        performance_results.to_csv(f'{csv_filename}_1.csv', index=False)
        train_proba_results.to_excel(f'{csv_filename}_2.xlsx', index=False)
        test_proba_results.to_excel(f'{csv_filename}_3.xlsx', index=False)
        y_train.to_csv(f'{csv_filename}_y_train.csv', index=False)
        pd.DataFrame(y_train_pred).to_csv(f'{csv_filename}_y_train_pred.csv', index=False)
        y_test1.to_csv(f'{csv_filename}_y_test.csv', index=False)
        pd.DataFrame(y_test_pred).to_csv(f'{csv_filename}_y_test_pred.csv', index=False)
        combined_train_results.to_csv(f'{csv_filename}_T.csv', index=False)
        combined_test_results.to_csv(f'{csv_filename}_v.csv', index=False)

