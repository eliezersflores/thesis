# -*- coding: utf-8 -*-

# This module contains the settings used in the experiments related with the classification step.

cnns = []
cnns.append('densenet121')
cnns.append('densenet169')
cnns.append('densenet201')
cnns.append('efficientnetb0')
cnns.append('efficientnetb1')
cnns.append('efficientnetb2')
cnns.append('efficientnetb3')
cnns.append('efficientnetb4')
cnns.append('efficientnetb5')
cnns.append('efficientnetb6')
cnns.append('efficientnetb7')
cnns.append('inceptionresnetv2')
cnns.append('inceptionv3')
cnns.append('mobilenet')
cnns.append('mobilenetv2')
cnns.append('mobilenetv3large')
cnns.append('mobilenetv3small')
cnns.append('nasnetlarge')
cnns.append('nasnetmobile')
cnns.append('resnet101')
cnns.append('resnet101v2')
cnns.append('resnet152')
cnns.append('resnet152v2')
cnns.append('resnet50')
cnns.append('resnet50v2')
cnns.append('vgg16')
cnns.append('vgg19')
cnns.append('xception')

datasets = ['DermIS', 'DermNet', 'DermQuest', 'MClass-ND']

labels = ['notmelanoma', 'melanoma']

#lambs = [0.01, 0.02, 0.03, 0.04, 0.05]
lambs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]

n_folds = 10

shapes = ['224', '240', '260', '299', '300', '331', '380', '456', '528', '600']

train_datasets = ['DermIS', 'DermQuest']

#vphis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
vphis = [0.2, 0.4, 0.6, 0.8, 1]


