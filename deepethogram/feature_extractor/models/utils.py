from torch import nn
import torch
import logging

log = logging.getLogger(__name__)


def pop(model, model_name, n_layers):
    # Goal of the pop function:
    # for each model, remove the N final layers.
    # Whatever permutation you make to the model, it MUST STILL USE THE MODEL'S OWN .FORWARD FUNCTION
    # Also goal: make it an intelligent number of layers
    # so when you pop off, for example, 1 layer from AlexNet
    # you also want to pop off the previous ReLU so that you get the unscaled linear units from fc_7
    # just doing something like model = nn.Sequential(*list(model.children())[:-1]) would not get rid of
    # this ReLU, so that's an unintelligent version of this
    if model_name.startswith('resnet'):
        if n_layers == 1:
            # use empty sequential module as an identity function
            num_features = model.fc.in_features
            final_layer = model.fc
            model.fc = nn.Identity()
        else:
            raise NotImplementedError('Can only pop off the final layer of a resnet')
    elif model_name == 'alexnet':
        final_layer = model.classifier
        if n_layers == 1:
            model.classifier = nn.Sequential(
                nn.Dropout(),
                # fc_6
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                # fc_7
                nn.Linear(4096, 4096),
            )
            num_features = 4096
            log.info('Final layer of encoder: AlexNet FC_7')
        elif n_layers == 2:
            model.classifier = nn.Sequential(
                nn.Dropout(),
                # fc_6
                nn.Linear(256 * 6 * 6, 4096),
            )
            num_features = 4096
            log.info('Final layer of encoder: AlexNet FC_6')
        elif n_layers == 3:
            # do nothing
            model.classifier = nn.Sequential()
            num_features = 256 * 6 * 6
            log.info('Final layer of encoder: AlexNet Maxpool 3')
        else:
            raise ValueError('Invalid parameter %d to pop function for %s: ' % (n_layers, model_name))

    elif model_name.startswith('vgg'):
        final_layer = model.classifier
        if n_layers == 1:
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
            )
            num_features = 4096
            log.info('Final layer of encoder: VGG fc2')
        elif n_layers == 2:
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
            )
            log.info('Final layer of encoder: VGG fc1')
            num_features = 4096
        elif n_layers == 3:
            model.classifier = nn.Sequential()
            log.info('Final layer of encoder: VGG pool5')
            num_features = 512 * 7 * 7
        else:
            raise ValueError('Invalid parameter %d to pop function for %s: ' % (n_layers, model_name))

    elif model_name.startswith('squeezenet'):
        raise NotImplementedError
    elif model_name.startswith('densenet'):
        raise NotImplementedError
    elif model_name.startswith('inception'):
        raise NotImplementedError
    else:
        raise ValueError('%s is not a valid model name' % (model_name))
    return model, num_features, final_layer


def remove_cnn_classifier_layer(cnn):
    """ Removes the final layer of a torchvision classification model, and figures out dimensionality of final layer """
    # cnn should be a nn.Sequential(custom_model, nn.Linear)
    module_list = list(cnn.children())
    assert (len(module_list) == 2 or len(module_list) == 3) and isinstance(module_list[1], nn.Linear)
    in_features = module_list[1].in_features
    module_list[1] = nn.Identity()
    cnn = nn.Sequential(*module_list)
    return cnn, in_features


# def replace_final_layers(model, model_name: str, num_classes: int):
#     model_classes = get_num_classes(model)
#     if num_classes == model_classes:
#         return model
#
#     if model_name.startswith('resnet'):
#         in_features = model.fc.in_features
#         # should happen when you want the final features of the model to be returned
#         if num_classes == 0:
#             model.fc = nn.Identity()
#         else:
#             model.fc = nn.Linear(in_features, num_classes)
#
#     elif model_name == 'alexnet':
#         in_features = model.classifier[6].in_features
#         if num_classes == 0:
#             model.classifier[6] = nn.Identity()
#         else:
#             model.classifier[6] = nn.Linear(in_features, num_classes)
#     elif model_name.startswith('vgg'):
#         in_features = model.classifier[6].in_features
#         if num_classes == 0:
#             model.classifier[6] = nn.Identity()
#         else:
#             model.classifier[6] = nn.Linear(in_features, num_classes)
#
#     elif model_name.startswith('squeezenet'):
#         if num_classes == 0:
#             model.classifier[1] = nn.Identity()
#         else:
#             model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
#             model.num_classes = num_classes
#
#     elif model_name.startswith('densenet'):
#         in_features = model.classifier.in_features
#         model.classifier = nn.Linear(in_features, num_classes)
#         if num_classes == 0:
#             model.classifier = nn.Identity()
#         else:
#             model.classifier = nn.Linear(in_features, num_classes)
#     elif model_name.startswith('inception'):
#         # handle auxiliary network
#         in_features = model.AuxLogits.fc.in_features
#         if num_classes == 0:
#             model.AuxLogits.fc = nn.Identity()
#         else:
#             model.AuxLogits.fc = nn.Linear(in_features, num_classes)
#         # handle primary network
#         in_features = model.fc.in_features
#         if num_classes == 0:
#             model.fc = nn.Identity()
#         else:
#             model.fc = nn.Linear(in_features, num_classes)
#     else:
#         raise ValueError('%s is not a valid model name' % (model_name))
#     return model


class Fusion(nn.Module):
    """ Module for fusing spatial and flow features and passing through Linear layer """

    def __init__(self, style, num_spatial_features, num_flow_features, num_classes, flow_fusion_weight=1.5,
                 activation=nn.Identity()):
        super().__init__()
        self.style = style
        self.num_classes = num_classes
        self.activation = activation
        self.flow_fusion_weight = flow_fusion_weight

        if self.style == 'average':
            # self.spatial_fc = nn.Linear(num_spatial_features,num_classes)
            # self.flow_fc = nn.Linear(num_flow_features, num_classes)

            self.num_features_out = num_classes

        elif self.style == 'concatenate':
            self.num_features_out = num_classes
            self.fc = nn.Linear(num_spatial_features + num_flow_features, num_classes)

        elif self.style == 'weighted_average':
            self.flow_weight = nn.Parameter(torch.Tensor([0.5]).float(), requires_grad=True)
        else:
            raise NotImplementedError

    def forward(self, spatial_features, flow_features):
        if self.style == 'average':
            # spatial_logits = self.spatial_fc(spatial_features)
            # flow_logits = self.flow_fc(flow_features)

            return (spatial_features + flow_features * self.flow_fusion_weight) / (1 + self.flow_fusion_weight)
            # return((spatial_logits+flow_logits*self.flow_fusion_weight)/(1+self.flow_fusion_weight))
        elif self.style == 'concatenate':
            # if we're concatenating, we want the model to learn nonlinear mappings from the spatial logits and flow
            # logits that means we should apply an activation function note: this won't work if you froze both
            # encoding models
            features = self.activation(torch.cat((spatial_features, flow_features), dim=1))
            return self.fc(features)
        elif self.style == 'weighted_average':
            return self.flow_weight*flow_features + (1-self.flow_weight)*spatial_features

# def get_num_classes(model):
#     classes = []
#
#     def get_linear_children(modulelist):
#         if len(modulelist) > 0:
#             for module in modulelist:
#                 if isinstance(module, torch.nn.Linear):
#                     classes.append(module.out_features)
#                 else:
#                     get_linear_children(list(module.children()))
#
#     get_linear_children(list(model.children()))
#     return classes


# from TResNet: https://arxiv.org/abs/2003.13630
# in my hands it was not appreciably faster
class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
