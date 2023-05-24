import dataclasses
import json
import os
import time
from dataclasses import dataclass
from glob import glob

from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.svm import OneClassSVM

import numpy as np
from tqdm import tqdm

from common import object_names


@dataclass
class Result:
    object_name: str
    augmented: bool
    feature_extraction_model_name: str
    feature_extraction_initialization: str
    feature_extraction_layer: str
    feature_extraction_shape: tuple[int, int]
    model_name: str
    hyperparameter: dict
    train_time: float | None
    test_inference_time: float | None
    auroc: float | None


def evaluate():
    train_paths = sorted(glob(f'data/*/feature_extraction/*train_good.npz'))
    for train_path in train_paths:
        path_split = train_path.split('/')
        file_split = path_split[-1].split('__')
        object_name = path_split[1]
        feature_extraction_model_name = file_split[0]
        feature_extraction_initialization = file_split[1]
        augmented = file_split[2] == 'aug'

        if object_name not in object_names:
            print(f'skipping: {object_name}')
            continue

        results: list[Result] = []
        result_path = f'results/{object_name}/{"__".join(file_split[:3])}.json'
        if os.path.exists(result_path):
            with open(result_path) as f:
                results = [Result(**j) for j in json.load(f)]
                if len(results) == 284:
                    continue
        with open(result_path, 'w') as f:
            json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

        print(f'working on dataset: {train_path}')
        train_data_npz = np.load(train_path)
        test_data_good_npz = np.load('/'.join(path_split[:-1]) + '/' + '__'.join(file_split[:2]) + '__test_good.npz')
        test_data_bad_npz = np.load('/'.join(path_split[:-1]) + '/' + '__'.join(file_split[:2]) + '__test_bad.npz')
        train_data = dict()
        test_data = dict()
        test_data_labels = dict()
        layers = ['layer0', 'layer1', 'layer2', 'layer3'] \
            if 'vit_base' in feature_extraction_model_name \
            else ['layer1', 'layer2', 'layer3', 'layer4']
        for layer in layers:
            train_data[layer] = train_data_npz[layer]
            test_data[layer] = np.concatenate((test_data_good_npz[layer], test_data_bad_npz[layer]))
            test_data_labels[layer] = np.concatenate(
                [
                    np.full([test_data_good_npz[layer].shape[0]], fill_value=1),
                    np.full([test_data_bad_npz[layer].shape[0]], fill_value=-1)
                ]
            )

        # One Class SVM
        model_name = 'sklearn.svm.OneClassSVM'
        grid = ParameterGrid(
            [
                {
                    'layer': layers,
                    'kernel': ['linear', 'rbf', 'sigmoid'],
                },
                {
                    'layer': layers,
                    'kernel': ['poly'],
                    'degree': [3, 5, 8, 13, 21, 34, 55]
                }
            ]
        )
        for params in tqdm(grid, desc='One Class SVM'):
            layer = params['layer']
            model_params = dict(params)
            del model_params['layer']

            result = Result(
                object_name=object_name,
                augmented=augmented,
                feature_extraction_model_name=feature_extraction_model_name,
                feature_extraction_initialization=feature_extraction_initialization,
                feature_extraction_layer=layer,
                feature_extraction_shape=train_data[layer].shape,
                model_name=model_name,
                hyperparameter=model_params,
                train_time=None,
                test_inference_time=None,
                auroc=None
            )
            if contains_result(results, result):
                continue
            results.append(result)
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

            start_time = time.time()
            clf = OneClassSVM(max_iter=1000, **model_params).fit(train_data[layer])
            train_time = time.time() - start_time

            start_time = time.time()
            test_data_predictions = clf.decision_function(test_data[layer])
            inference_time = time.time() - start_time

            test_data_predictions = np.nan_to_num(test_data_predictions)
            auroc = roc_auc_score(test_data_labels[layer], test_data_predictions)

            result.train_time = train_time
            result.test_inference_time = inference_time
            result.auroc = auroc
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

        # Isolation Forest
        model_name = 'sklearn.ensemble.IsolationForest'
        grid = ParameterGrid(
            [
                {
                    'layer': layers,
                    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                }
            ]
        )
        for params in tqdm(grid, desc='Isolation Forest'):
            layer = params['layer']
            model_params = dict(params)
            del model_params['layer']

            result = Result(
                object_name=object_name,
                augmented=augmented,
                feature_extraction_model_name=feature_extraction_model_name,
                feature_extraction_initialization=feature_extraction_initialization,
                feature_extraction_layer=layer,
                feature_extraction_shape=train_data[layer].shape,
                model_name=model_name,
                hyperparameter=model_params,
                train_time=None,
                test_inference_time=None,
                auroc=None
            )
            if contains_result(results, result):
                continue
            results.append(result)
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

            start_time = time.time()
            clf = IsolationForest(**model_params).fit(train_data[layer])
            train_time = time.time() - start_time

            start_time = time.time()
            test_data_predictions = clf.decision_function(test_data[layer])
            inference_time = time.time() - start_time

            test_data_predictions = np.nan_to_num(test_data_predictions)
            auroc = roc_auc_score(test_data_labels[layer], test_data_predictions)

            result.train_time = train_time
            result.test_inference_time = inference_time
            result.auroc = auroc
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

        # Kernel Density Estimation
        model_name = 'sklearn.neighbors.KernelDensity'
        grid = ParameterGrid(
            [
                {
                    'layer': layers,
                    'bandwidth': [1.0, 0.5, 'scott', 'silverman'],
                    'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
                }
            ]
        )
        for params in tqdm(grid, desc='Kernel Density Estimation'):
            layer = params['layer']
            model_params = dict(params)
            del model_params['layer']

            result = Result(
                object_name=object_name,
                augmented=augmented,
                feature_extraction_model_name=feature_extraction_model_name,
                feature_extraction_initialization=feature_extraction_initialization,
                feature_extraction_layer=layer,
                feature_extraction_shape=train_data[layer].shape,
                model_name=model_name,
                hyperparameter=model_params,
                train_time=None,
                test_inference_time=None,
                auroc=None
            )
            if contains_result(results, result):
                continue
            results.append(result)
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

            start_time = time.time()
            clf = KernelDensity(**model_params).fit(train_data[layer])
            train_time = time.time() - start_time

            start_time = time.time()
            test_data_predictions = clf.score_samples(test_data[layer])
            inference_time = time.time() - start_time

            test_data_predictions = np.nan_to_num(test_data_predictions)
            auroc = roc_auc_score(test_data_labels[layer], test_data_predictions)

            result.train_time = train_time
            result.test_inference_time = inference_time
            result.auroc = auroc
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

        # Local Outlier Factor
        model_name = 'sklearn.neighbors.LocalOutlierFactor'
        grid = ParameterGrid(
            [
                {
                    'layer': layers,
                    'n_neighbors': [2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64],
                }
            ]
        )
        for params in tqdm(grid, desc='Local Outlier Factor'):
            layer = params['layer']
            model_params = dict(params)
            del model_params['layer']

            result = Result(
                object_name=object_name,
                augmented=augmented,
                feature_extraction_model_name=feature_extraction_model_name,
                feature_extraction_initialization=feature_extraction_initialization,
                feature_extraction_layer=layer,
                feature_extraction_shape=train_data[layer].shape,
                model_name=model_name,
                hyperparameter=model_params,
                train_time=None,
                test_inference_time=None,
                auroc=None
            )
            if contains_result(results, result):
                continue
            results.append(result)
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

            start_time = time.time()
            clf = LocalOutlierFactor(novelty=True, **model_params).fit(train_data[layer])
            train_time = time.time() - start_time

            start_time = time.time()
            test_data_predictions = clf.decision_function(test_data[layer])
            inference_time = time.time() - start_time

            test_data_predictions = np.nan_to_num(test_data_predictions)
            auroc = roc_auc_score(test_data_labels[layer], test_data_predictions)

            result.train_time = train_time
            result.test_inference_time = inference_time
            result.auroc = auroc
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

        # Elliptic Envelope
        model_name = 'sklearn.covariance.EllipticEnvelope'
        grid = ParameterGrid(
            [
                {
                    'layer': layers,
                    'dim_reduction': ['pca'],
                    'dim_reduction_components': [32, 64, 128, 256],
                }
            ]
        )
        for params in tqdm(grid, desc='Elliptic Envelope'):
            layer = params['layer']
            del params['layer']

            result = Result(
                object_name=object_name,
                augmented=augmented,
                feature_extraction_model_name=feature_extraction_model_name,
                feature_extraction_initialization=feature_extraction_initialization,
                feature_extraction_layer=layer,
                feature_extraction_shape=train_data[layer].shape,
                model_name=model_name,
                hyperparameter=params,
                train_time=None,
                test_inference_time=None,
                auroc=None
            )
            if contains_result(results, result):
                continue
            results.append(result)
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

            dim_reduction = params['dim_reduction']
            dim_reduction_components = params['dim_reduction_components']
            if dim_reduction == 'pca':
                if dim_reduction_components > min(train_data[layer].shape[0], train_data[layer].shape[1]):
                    dim_reduction_components = None
                pca = PCA(n_components=dim_reduction_components).fit(train_data[layer])
                train_data_dim_reduction = pca.transform(train_data[layer])
                test_data_dim_reduction = pca.transform(test_data[layer])
            else:
                raise ValueError(f'illegal {dim_reduction=}')

            start_time = time.time()
            clf = EllipticEnvelope().fit(train_data_dim_reduction)
            train_time = time.time() - start_time

            start_time = time.time()
            test_data_predictions = clf.decision_function(test_data_dim_reduction)
            inference_time = time.time() - start_time

            test_data_predictions = np.nan_to_num(test_data_predictions)
            auroc = roc_auc_score(test_data_labels[layer], test_data_predictions)

            result.train_time = train_time
            result.test_inference_time = inference_time
            result.auroc = auroc
            with open(result_path, 'w') as f:
                json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)

        with open(result_path, 'w') as f:
            json.dump([dataclasses.asdict(r) for r in results], f, indent=2, sort_keys=True)


def contains_result(results: list[Result], result: Result):
    for other in results:
        if result.object_name == other.object_name and \
                result.augmented == other.augmented and \
                result.feature_extraction_model_name == other.feature_extraction_model_name and \
                result.feature_extraction_initialization == other.feature_extraction_initialization and \
                result.feature_extraction_layer == other.feature_extraction_layer and \
                result.model_name == other.model_name and \
                result.hyperparameter == other.hyperparameter:
            return True
    return False


if __name__ == '__main__':
    evaluate()
