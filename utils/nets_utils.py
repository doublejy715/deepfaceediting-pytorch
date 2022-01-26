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

def get_num_adain_params(model):
        # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

def assign_adain_params(model, adain_params):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean
            m.weight = std
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]
