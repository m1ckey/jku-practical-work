import csv
import json
import glob

import pandas as pd
import matplotlib.pyplot as plt

from common import object_names


results: list[dict] = []
result_paths = glob.glob('results/*/*.json')
assert len(result_paths) == 240
for path in result_paths:
    with open(path) as f:
        j = json.load(f)
        assert len(j) == 284
        results += j

'''
15 datasets
4 feature models
2 feature model initializations
2 augmentations
4 layers
= 960 training datasets

one class svm: 10 hyperparams
isolation forest: 19 hyperparameters
kernel density estimation: 24 hyperparameters
local outlier factor: 14 hyperparameters
elliptic envelope: 4 hyperparameters (pca)
= 71 models/hyperparameters

= 68160 experiments
'''
assert len(results) == 68160

failed_experiments = [
    result
    for result in results
    if result['auroc'] is None
]
print(f'experiments: {len(results)} ({len(failed_experiments)} failed)')

with open('results/results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'object_name',
            'feature_extraction_model_name',
            'feature_extraction_initialization',
            'augmented',
            'feature_extraction_layer',
            'feature_extraction_shape',
            'model_name',
            'hyperparameter',
            'auroc',
            'train_time',
            'test_inference_time',
        ]
    )
    writer.writeheader()
    writer.writerows(results)

top_1 = {}
top_10 = {}
for object_name in object_names:
    object_results = [
        result
        for result in results
        if result['object_name'] == object_name
        and result['auroc'] is not None
    ]
    object_results.sort(key=lambda r: r['auroc'] if r['auroc'] else 0, reverse=True)
    top_1[object_name] = object_results[0]
    top_10[object_name] = object_results[:10]

average_auroc = 0
for object_name in object_names:
    average_auroc += top_1[object_name]['auroc']
average_auroc /= len(object_names)
print(f'avg auroc: {average_auroc}')

with open('results/results_top_1.csv', 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'object_name',
            'feature_extraction_model_name',
            'feature_extraction_initialization',
            'augmented',
            'feature_extraction_layer',
            'feature_extraction_shape',
            'model_name',
            'hyperparameter',
            'auroc',
            'train_time',
            'test_inference_time',
        ]
    )
    writer.writeheader()
    for object_name in object_names:
        writer.writerow(top_1[object_name])

with open('results/results_top_10.csv', 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            'object_name',
            'feature_extraction_model_name',
            'feature_extraction_initialization',
            'augmented',
            'feature_extraction_layer',
            'feature_extraction_shape',
            'model_name',
            'hyperparameter',
            'auroc',
            'train_time',
            'test_inference_time',
        ]
    )
    writer.writeheader()
    for object_name in object_names:
        writer.writerows(top_10[object_name])


'''
select  model, vision_model_layer_shape_dim, avg(train_time), avg(test_time), count(*)
from result
group by model, vision_model_layer_shape_dim
'''

'''
select  model, vision_model_layer_shape_samples, avg(train_time), avg(test_time), count(*)
from result
group by model, vision_model_layer_shape_samples
'''


data_dims = pd.read_csv('results/avg_compute_time_dims.csv')
data_samples = pd.read_csv('results/avg_compute_time_samples.csv')

models_dims = data_dims['model']
dims_dims = data_dims['dims']
train_times_dims = data_dims['train time']
test_times_dims = data_dims['test time']

models_samples = data_samples['model']
dims_samples = data_samples['samples']
train_times_samples = data_samples['train time']
test_times_samples = data_samples['test time']

plt.figure(dpi=150, figsize=(10, 6))

ax1 = plt.subplot(211)
unique_models_dims = list(set(models_dims))
colors = plt.cm.tab10([i / len(unique_models_dims) for i in range(len(unique_models_dims))])

for i, model in enumerate(unique_models_dims):
    model_dims = dims_dims[models_dims == model]
    model_train_times = train_times_dims[models_dims == model]
    model_test_times = test_times_dims[models_dims == model]

    ax1.plot(model_dims, model_train_times, color=colors[i], label=model + ' - Train')
    ax1.plot(model_dims, model_test_times, color=colors[i], linestyle='--', label=model + ' - Test')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Sample dimension')
ax1.set_ylabel('Time [s]')
ax1.set_title('Compute Time - Sample dimension')
ax1.legend(bbox_to_anchor=(1.05, 1))

ax2 = plt.subplot(212)
unique_models_samples = list(set(models_samples))

for i, model in enumerate(unique_models_samples):
    model_dims = dims_samples[models_samples == model]
    model_train_times = train_times_samples[models_samples == model]
    model_test_times = test_times_samples[models_samples == model]

    ax2.plot(model_dims, model_train_times, color=colors[i], label=model + ' - Train')
    ax2.plot(model_dims, model_test_times, color=colors[i], linestyle='--', label=model + ' - Test')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Time [s]')
ax2.set_title('Compute Time - Samples')

plt.tight_layout()
plt.show()