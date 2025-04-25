import json
import pandas as pd

# 1. Load JSON
with open('results/all_results.json', 'r') as f:
    data = json.load(f)

rows = []

for mdl in data:
    name = mdl['model']
    is_pt = isinstance(mdl['attempts'][0]['train'], list)  # Pytorch model has list in 'train'
    
    # nếu baseline (not list), chỉ có một bộ train/valid/tests
    if not is_pt:
        # tính trung bình train_time và single_predict_time
        avg_train_time = sum(a['train_time'] for a in mdl['attempts']) / mdl['num_attempts']
        avg_sp_time    = sum(a['single_predict_time'] for a in mdl['attempts']) / mdl['num_attempts']
        # trung bình MSE/MAE/DA trên train, valid; chọn tests[0]
        for split in ['train','valid']:
            agg = {}
            for metric in ['mse','mae','da']:
                # mỗi attempt có mdl['attempts'][i][split][metric]['last']
                avg = sum(a[split][metric]['last'] for a in mdl['attempts']) / mdl['num_attempts']
                agg[metric] = avg
            rows.append({
                'Model': name,
                'Split': split.capitalize(),
                'MSE': agg['mse'],
                'MAE': agg['mae'],
                'DA':  agg['da'],
                'Train Time': avg_train_time,
                'Single Predict Time': avg_sp_time,
            })
        # tests: chỉ test_id=0
        agg = {}
        for metric in ['mse','mae','da']:
            avg = sum(a['tests'][0][metric]['last'] for a in mdl['attempts']) / mdl['num_attempts']
            agg[metric] = avg
        rows.append({
            'Model': name,
            'Split': 'Test',
            'MSE': agg['mse'],
            'MAE': agg['mae'],
            'DA':  agg['da'],
            'Train Time': avg_train_time,
            'Single Predict Time': avg_sp_time,
        })

    else:
        # với Pytorch: tách last và best
        avg_train_time = sum(a['train_time'] for a in mdl['attempts']) / mdl['num_attempts']
        avg_sp_time    = sum(a['single_predict_time'] for a in mdl['attempts']) / mdl['num_attempts']
        for typ in ['last','best']:
            for split in ['train','valid']:
                # mỗi attempt a['train'] là list 2 dicts, filter by type
                agg = {}
                for metric in ['mse','mae','da']:
                    vals = []
                    for a in mdl['attempts']:
                        block = next(x for x in a[split] if x['type']==typ)
                        vals.append(block[metric]['last'])
                    agg[metric] = sum(vals)/len(vals)
                rows.append({
                    'Model': f"{name}-{typ}",
                    'Split': split.capitalize(),
                    'MSE': agg['mse'],
                    'MAE': agg['mae'],
                    'DA':  agg['da'],
                    'Train Time': avg_train_time,
                    'Single Predict Time': avg_sp_time,
                })
            # tests (chỉ test_id=0)
            agg = {}
            for metric in ['mse','mae','da']:
                vals = []
                for a in mdl['attempts']:
                    tests_block = next(tb for tb in a['tests'] if tb['type']==typ)
                    # tests_block['tests'] is list; chọn test_id==0
                    tb0 = next(t for t in tests_block['tests'] if t['test_id']==0)
                    vals.append(tb0[metric]['last'])
                agg[metric] = sum(vals)/len(vals)
            rows.append({
                'Model': f"{name}-{typ}",
                'Split': 'Test',
                'MSE': agg['mse'],
                'MAE': agg['mae'],
                'DA':  agg['da'],
                'Train Time': avg_train_time,
                'Single Predict Time': avg_sp_time,
            })

# 2. Build DataFrame và pivot (nếu cần)
df = pd.DataFrame(rows)

# ví dụ pivot để ra cột Train/Valid/Test như ảnh
table = df.pivot_table(
    index='Model',
    columns='Split',
    values=['MSE','MAE','DA'],
    aggfunc='first'
)

# flatten multiindex
table.columns = [f"{metric}_{split}" for metric, split in table.columns]

# thêm 2 cột thời gian
times = df[['Model','Train Time','Single Predict Time']].drop_duplicates().set_index('Model')
final = table.join(times)

final.to_csv('results/analysis_last.csv', index=True)
