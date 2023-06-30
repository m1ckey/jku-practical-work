import csv
import json
import glob

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
