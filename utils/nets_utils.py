import functools
import torch
import torch.nn as nn

part = {'mouth': (169, 301, 192, 192),
        'nose': (182, 232, 160, 160-36),
        'eye1': (108, 156, 128, 128),
        'eye2': (255, 156, 128, 128)}
face_components = part.keys()

def combine_feature_map(part_features):
    feature_map = part_features['bg']
    for component in face_components:
        x,y,width,height = component
        feature_map[:, :, x:x+width, y:y+height] = part_features[component]

    return feature_map

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
