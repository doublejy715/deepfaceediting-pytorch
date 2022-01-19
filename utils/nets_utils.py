from nets.encoder import Style_Encoder

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

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

def get_generated_part_feature(model, image_content, image_style):
    num_adain_params =get_num_adain_params(model)
    style_encoder = Style_Encoder(input_cn=3,style_dim=num_adain_params)
    adain_params = style_encoder(image_style)

    assign_adain_params(adain_params, model)
    for layer_id, layer in enumerate(model):
        image_content = layer(image_content)
        if layer_id == 15:
            return image_content

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean
            m.weight = std
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

def get_part_feature(model, image_content, image_style, adain_params = None):
    num_adain_params =get_num_adain_params(model)
    style_encoder = Style_Encoder(input_cn=3,style_dim=num_adain_params)
    adain_params = style_encoder(image_style)
    #print(adain_params.shape)
    assign_adain_params(adain_params, model)
    return model(image_content),adain_params