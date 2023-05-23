import os
from glob import glob
from sys import argv
from typing import Callable
import itertools

import numpy as np
import torch
from torch import nn
from torchvision.io import read_image, ImageReadMode
from torchvision.models import \
    resnet18, \
    ResNet18_Weights, \
    resnet50, \
    ResNet50_Weights, \
    vit_b_16, \
    ViT_B_16_Weights, \
    vit_b_32, \
    ViT_B_32_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from common import object_names

for object_name in object_names:
    assert object_name in os.listdir('./data')


def extract_features_resnet18():
    with torch.no_grad():
        for weights, augmented in itertools.product(
                [None, ResNet18_Weights.DEFAULT],
                [True, False],
        ):
            name = f'resnet18__{weights.name.lower() if weights else "random"}'
            model = resnet18(
                weights=weights
            )
            model.eval()
            run_and_save_all(
                name=name,
                augmented=augmented,
                feature_extractor=create_feature_extractor(
                    model=model,
                    return_nodes=['layer1', 'layer2', 'layer3', 'layer4']
                ),
                preprocess=ResNet18_Weights.DEFAULT.transforms()
            )


def extract_features_resnet50():
    with torch.no_grad():
        for weights, augmented in itertools.product(
                [None, ResNet50_Weights.DEFAULT],
                [True, False],
        ):
            name = f'resnet50__{weights.name.lower() if weights else "random"}'
            model = resnet50(
                weights=weights
            )
            model.eval()
            run_and_save_all(
                name=name,
                augmented=augmented,
                feature_extractor=create_feature_extractor(
                    model=model,
                    return_nodes=['layer1', 'layer2', 'layer3', 'layer4']
                ),
                preprocess=ResNet50_Weights.DEFAULT.transforms()
            )


def extract_features_vit_base_16():
    with torch.no_grad():
        for weights, augmented in itertools.product(
                [None, ViT_B_16_Weights.DEFAULT],
                [True, False],
        ):
            name = f'vit_base_16__{weights.name.lower() if weights else "random"}'
            model = vit_b_16(
                weights=weights
            )
            model.eval()
            run_and_save_all(
                name=name,
                augmented=augmented,
                feature_extractor=create_feature_extractor(
                    model=model,
                    return_nodes={
                        'encoder.layers.encoder_layer_0': 'layer0',
                        'encoder.layers.encoder_layer_1': 'layer1',
                        'encoder.layers.encoder_layer_2': 'layer2',
                        'encoder.layers.encoder_layer_3': 'layer3',
                    }
                ),
                preprocess=ViT_B_16_Weights.DEFAULT.transforms()
            )


def extract_features_vit_base_32():
    with torch.no_grad():
        for weights, augmented in itertools.product(
                [None, ViT_B_32_Weights.DEFAULT],
                [True, False],
        ):
            name = f'vit_base_32__{weights.name.lower() if weights else "random"}'
            model = vit_b_32(
                weights=weights
            )
            model.eval()
            run_and_save_all(
                name=name,
                augmented=augmented,
                feature_extractor=create_feature_extractor(
                    model=model,
                    return_nodes={
                        'encoder.layers.encoder_layer_0': 'layer0',
                        'encoder.layers.encoder_layer_1': 'layer1',
                        'encoder.layers.encoder_layer_2': 'layer2',
                        'encoder.layers.encoder_layer_3': 'layer3',
                    }
                ),
                preprocess=ViT_B_32_Weights.DEFAULT.transforms()
            )


def run_and_save_all(
        name: str,
        augmented: bool,
        feature_extractor: nn.Module,
        preprocess: Callable
):
    print(f'extracting {"augmented" if augmented else "non-augmented"} features with {name}:')

    for object_name in object_names:
        print(f'{object_name} ', end='')
        train_good = glob(f'./data/{object_name}/train/good/*.png')
        if augmented:
            train_good += glob(f'./data/{object_name}/train/good/augmented/*.png')
        test_good = glob(f'./data/{object_name}/test/good/*.png')
        test_bad = list(set(glob(f'./data/{object_name}/test/*/*.png')) - set(test_good))

        run_and_save(
            name=f'{name}__{"aug" if augmented else "noaug"}__train_good',
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            object_name=object_name,
            paths=train_good
        )
        run_and_save(
            name=f'{name}__test_good',
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            object_name=object_name,
            paths=test_good
        )
        run_and_save(
            name=f'{name}__test_bad',
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            object_name=object_name,
            paths=test_bad
        )

    print()


def run_and_save(
        name: str,
        feature_extractor: nn.Module,
        preprocess: Callable,
        object_name: str,
        paths: list[str],
):
    save_path = f'./data/{object_name}/feature_extraction/{name}.npz'
    if os.path.exists(save_path):
        return

    features = []
    for path in paths:
        features.append(
            feature_extractor(
                preprocess(
                    read_image(path, ImageReadMode.RGB)
                )
                .unsqueeze(0)
            )
        )
    layers = {}
    for layer_name in features[0].keys():
        layers[layer_name] = np.stack([f[layer_name].flatten() for f in features])

    if not os.path.exists(f'./data/{object_name}/feature_extraction/'):
        os.makedirs(f'./data/{object_name}/feature_extraction/')
    np.savez_compressed(save_path, **layers)


if __name__ == '__main__':
    extract_features_resnet18()
    extract_features_resnet50()
    extract_features_vit_base_16()
    extract_features_vit_base_32()

