import os
from glob import glob
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
    ViT_B_16_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from common import object_names

for object_name in object_names:
    assert object_name in os.listdir('./data')


def preprocess():
    with torch.no_grad():
        # resnet18
        for weights, augment in itertools.product(
                [None, ResNet18_Weights.DEFAULT],
                [False],
        ):
            name = f'resnet18__{weights.name.lower() if weights else "random"}'
            model = resnet18(
                weights=weights
            )
            model.eval()
            extract_features_all(
                name=name,
                feature_extractor=create_feature_extractor(
                    model=model,
                    return_nodes=['layer1', 'layer2', 'layer3', 'layer4']
                ),
                preprocess=ResNet18_Weights.DEFAULT.transforms()
            )

        # resnet50
        for weights, augment in itertools.product(
                [None, ResNet50_Weights.DEFAULT],
                [False],
        ):
            name = f'resnet50__{weights.name.lower() if weights else "random"}'
            model = resnet50(
                weights=weights
            )
            model.eval()
            extract_features_all(
                name=name,
                feature_extractor=create_feature_extractor(
                    model=model,
                    return_nodes=['layer1', 'layer2', 'layer3', 'layer4']
                ),
                preprocess=ResNet50_Weights.DEFAULT.transforms()
            )

        # vit_base_16
        for weights, augment in itertools.product(
                [None, ViT_B_16_Weights.DEFAULT],
                [False],
        ):
            name = f'vit_base_16__{weights.name.lower() if weights else "random"}'
            model = vit_b_16(
                weights=weights
            )
            model.eval()
            extract_features_all(
                name=name,
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


def extract_features_all(
        name: str,
        feature_extractor: nn.Module,
        preprocess: Callable
):
    print(f'extracting features with {name}:')

    for object_name in object_names:
        print(f'{object_name} ', end='')
        train_good = glob(f'./data/{object_name}/train/good/*.png')
        test_good = glob(f'./data/{object_name}/test/good/*.png')
        test_bad = list(set(glob(f'./data/{object_name}/test/*/*.png')) - set(test_good))

        extract_features(
            name=f'{name}__train_good',
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            object_name=object_name,
            paths=train_good
        )
        extract_features(
            name=f'{name}__test_good',
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            object_name=object_name,
            paths=test_good
        )
        extract_features(
            name=f'{name}__test_bad',
            feature_extractor=feature_extractor,
            preprocess=preprocess,
            object_name=object_name,
            paths=test_bad
        )

    print()


def extract_features(
        name: str,
        feature_extractor: nn.Module,
        preprocess: Callable,
        object_name: str,
        paths: list[str],
):
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

    np.savez_compressed(f'./data/{object_name}/{name}.npz', **layers)


if __name__ == '__main__':
    preprocess()
