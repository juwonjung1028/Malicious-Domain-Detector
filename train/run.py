"""Run experiments and create figs"""
import itertools
import os
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

import lstm as lstm

from numpy import interp
from sklearn.metrics import roc_curve, auc

RESULT_FILE = './results.pkl'

def run_experiments(nfolds=5):
    model_results = {
        'lstm': None,
    }

    options = {
        'nfolds': nfolds,
        # enable for quick functional testing
        'max_epoch': 10
    }

#     # Add LSTM cross-validation
#     if 'lstm' in model_results:
#         x, y = get_data()
#         skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
#         fold_results = []
#         for train_index, test_index in skf.split(x, y):
#             x_train, x_test = x[train_index], x[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#             lstm_result = lstm.run(x_train, y_train, x_test, y_test, **options)
#             fold_results.append(lstm_result)
#         model_results['lstm'] = fold_results
        
    """Runs all experiments"""

    print ('==============   MALICIOUS DOMAIN DETERMINATION BASED ON LSTM MODEL   ==============\n')
    try:
        model_results['lstm'] = lstm.run(**options)
        
    except Exception as e:
        print("lstm error:", e)

    return {
        'options': options,
        'model_results': model_results,
    }

def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)

def calculate_metrics(model_results):

    fpr = []
    tpr = []
    for model_result in model_results:
        if model_result is not None:
            t_fpr, t_tpr, _ = roc_curve(model_result['score'], model_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
    model_fpr, model_tpr, model_auc = calc_macro_roc(fpr, tpr)
    return model_fpr, model_tpr, model_auc


# def calculate_metrics(model_results):
#     fpr = []
#     tpr = []
#     best_aucs = []  # 최고 auc 값을 저장할 리스트
#     for model_result in model_results:
#         if model_result is not None:
#             t_fpr, t_tpr, _ = roc_curve(model_result['y'], model_result['probs'])
#             fpr.append(t_fpr)
#             tpr.append(t_tpr)
#             _, _, model_auc = calc_macro_roc(fpr, tpr)
#             best_aucs.append(np.max(model_auc))  # 최고 auc 값을 저장
#     model_fpr, model_tpr, model_auc = calc_macro_roc(fpr, tpr)
#     mean_auc = np.mean(best_aucs)  # 최고 auc 값들의 평균값 계산
#     return model_fpr, model_tpr, mean_auc

# def calculate_metrics(model_results):
#     fpr = []
#     tpr = []
#     aucs = []  # auc 값을 저장할 리스트
#     for model_result in model_results:
#         if model_result is not None:
#             t_fpr, t_tpr, _ = roc_curve(model_result['y'], model_result['probs'])
#             fpr.append(t_fpr)
#             tpr.append(t_tpr)
#             _, mean_tpr, _ = calc_macro_roc(fpr, tpr)
#             aucs.append(auc(fpr, mean_tpr))  # auc 값을 저장
#     model_fpr, model_tpr, mean_auc = calc_macro_roc(fpr, tpr)
#     return model_fpr, model_tpr, np.mean(aucs)  # 평균 auc 값 계산



def create_figs(nfolds=5, force=False):
    """Create figures"""
    # Generate results if needed
    

    
    if force or (not os.path.isfile(RESULT_FILE)):
        results = run_experiments(nfolds=nfolds)
        with open(RESULT_FILE, 'wb') as file:
            pickle.dump(results, file)
        
        
        ################ Jason 파일로 또 볼수 있게 설정 ################
        
#         with open(RESULT_FILE, 'rb') as file:
#             data = pickle.load(file)

#         # 결과 객체를 JSON 형식으로 변환
#         # NumPy 배열을 리스트로 변환하여 직렬화
#         for model_name, model_result in data['model_results'].items():
#             if model_result is not None:
#                 data['model_results'][model_name]['fpr'] = data['model_results'][model_name]['fpr'].tolist()
#                 data['model_results'][model_name]['tpr'] = data['model_results'][model_name]['tpr'].tolist()
                
#         json_data = json.dumps(data, indent=4)  # indent는 가독성을 위한 들여쓰기를 추가하는 옵션

#         # JSON 데이터를 텍스트 파일에 저장
#         with open('results.json', 'w') as json_file:
#             json_file.write(json_data)
        
        
        ################################################################
            
            
            
    else:
        with open(RESULT_FILE, 'rb') as file:
            results = pickle.load(file)
            
    
    print("\n\n\n=========================   TEST COMPLETED SUCCESSFULLY!   =========================\n\n\n\n")
    
    lstm_results = results['model_results']['lstm']  # Get the list of results for LSTM
  
    print('====================================================================================')
    print('=================================   THE RESULTS   ==================================')
    print('====================================================================================\n')
    print(f"Number of Folds: {nfolds}")  # Print nfolds
    print(f"Max Epoch: {results['options']['max_epoch']}")  # Print max_epoch
            
    # Extract the 'score', 'labels', 'probs', 'confusion_matrix', and 'confusion_matrix_percent' data from the results

    if lstm_results is not None:
        for i, lstm_result in enumerate(lstm_results):
                      
            print("\n------------------------------------------------------------------------------------")
            print(f"\n[Fold {i + 1}] OUTPUT DATA (First 10 ~ Last 10):")
            
            print("\n- LABEL:\n", lstm_result['labels'][:10], " ~\n", lstm_result['labels'][-10:])
            print("\n- SCORE:\n", lstm_result['score'][:10], " ~\n", lstm_result['score'][-10:])
            print("\n- PROBABILITIES:\n", lstm_result['probs'][:10], " ~\n\n", lstm_result['probs'][-10:])
            
            print("\n- CONFUSION MATRIX(val):")
            print(lstm_result['confusion_matrix'])
            
            print("\n- CONFUSION MATRIX(%):")
            print(lstm_result['confusion_matrix_percent'])
    

            
    metrics = []
    for name, model_result in sorted(results['model_results'].items()):
        if model_result is not None:
                                    
            fpr, tpr, auc = calculate_metrics(model_result)
            
#             fpr_mean = np.mean(fpr, axis=0)  # FPR 값들의 평균 계산
#             tpr_mean = np.mean(tpr, axis=0)  # TPR 값들의 평균 계산
#             auc_mean = np.mean(auc) # AUC 값들의 평균 계산
#             print("Mean AUC for %s: %.4f" % (name, auc_mean))
            
            print("\n\n\n")
            print('====================================================================================\n')
            print("                           최종 %s 모델 AUC 값: %.4f"%(name, auc))
            print('\n====================================================================================')
            print('\n\nFin.')

            metrics.append({
                'name': name,
#                 'fpr': fpr_mean,  # 평균 FPR 값 사용
#                 'tpr': tpr_mean,  # 평균 TPR 값 사용
#                 'auc': auc_mean,  # 평균 AUC 값 사용
                
                'fpr': fpr,
                'tpr': tpr,                
                'auc': auc,
            })

    colors = {
        'lstm': 'blue',
    }

    # these control which models get grouped together when ROC images get created.
    metrics_filters = {
        'lstm': ('lstm',),
    }

    # Save figures
    with plt.style.context('bmh'):
        for plot_name, metrics_filter in metrics_filters.items():
            fig1, ax = plt.subplots()

            metrics_to_plot = [rec for rec in metrics if rec['name'] in metrics_filter]
            artists = []
            for metrics_record in sorted(metrics_to_plot, key=lambda rec: rec['auc'], reverse=True):

                linestyle = '-'
#                 if 'aloha' in metrics_record['name']:
#                     linestyle = ':'

#                 color = colors[metrics_record['name'].replace('aloha_', '')]
                color = colors[metrics_record['name']]
                
                label = '%s (AUC = %.4f)' % (metrics_record['name'].upper(), metrics_record['auc'])
                if not label.startswith('_'):  # Ignore labels starting with '_'
                    artist, = ax.plot(
                        metrics_record['fpr'],
                        metrics_record['tpr'],
                        label=label,
                        rasterized=True,
                        linestyle=linestyle,
                        color=color,
                    )
                    artists.append(artist)

            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title('ROC - Summary' , fontsize=20)
            
            if artists:  # 레전드에 아티스트가 있는 경우에만 호출.
                ax.legend(loc="best", fontsize=10)
                
            ax.tick_params(axis='both', labelsize=10)

            # create ROC curves at various zooms at linear scale
            ax.set_xscale('linear')
            for xmax in [0.1]:
                ax.set_xlim([0.0, xmax + 0.001])
                fig1.savefig('results-linear-{plot_name}-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

            
            
#             for xmax in [0.2, 0.4, 0.6, 0.8, 1.0]:
#                 ax.set_xlim([0.0, xmax + 0.001])
#                 fig1.savefig('results-linear-{plot_name}-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

            # create ROC curves at various zooms at log scale
            ax.set_xscale('log')
            for xmax in [0.2]:
                ax.set_xlim([0.000001, xmax + 0.001])
                fig1.savefig('results-logscale-{plot_name}-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

            
            
#             for xmax in [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#                 ax.set_xlim([0.000001, xmax + 0.001])
#                 fig1.savefig('results-logscale-{plot_name}-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

#             for xmax in [1.05]:
#                 ax.set_xlim([0.000001, xmax + 0.001])
#                 fig1.savefig('results-logscale-{plot_name}-0.000001-to-{xmax}.png'.format(xmax=xmax, plot_name=plot_name), pad_inches=0.25, bbox_inches='tight')

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    create_figs(nfolds=5) # Run with 1 to make it fast