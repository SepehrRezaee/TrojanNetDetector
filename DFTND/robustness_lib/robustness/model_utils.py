import torch as ch
import dill
import os
from . import helpers
from .attacker import AttackerModel
from . import imagenet_models as models

def make_and_restore_model(*_, arch, dataset, resume_path=None, 
        state_dict_path='model', resume_epoch=None, parallel=True):
    """
    make_and_restore_model
    Makes a model and (optionally) restores it from a checkpoint
    - arch (str): Model architecture identifier
    - dataset (Dataset class [see datasets.py])
    - resume_path (str): optional path to checkpoint

    Returns: model (possible loaded with checkpoint), checkpoint
    """
    classifier_model = dataset.get_model(arch)
    classifier_model.load_state_dict(ch.load(resume_path)['model'])

    # print(classifier_model)

    model = AttackerModel(classifier_model, dataset)

    # print(model.items())

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = ch.load(resume_path)['model']

            model = model.cuda()

        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint
