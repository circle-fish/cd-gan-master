import os

from code.config import Config


def make_if_not(file):
    if not os.path.exists(file):
        os.mkdir(file)


def do_common(config: Config, direction):
    config.model_dir = "{}/model".format(config.super_base)
    config.classifier_dir = "{}/classifier".format(config.model_dir)
    config.direction_dir = "{}/{}".format(config.model_dir, direction)
    config.result_dir = "{}/result".format(config.direction_dir)
    print('config.super_base:', config.super_base)
    make_if_not(config.super_base)
    make_if_not(config.model_dir)
    make_if_not(config.classifier_dir)
    make_if_not(config.direction_dir)
    make_if_not(config.result_dir)
