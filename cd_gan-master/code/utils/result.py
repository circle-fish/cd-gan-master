import numpy as np
import os.path as path
import pandas as pd
import os
import torch

from code.config import Config
from code.trainer.GanTrainer5 import GanTrainer5
from code.trainer.SingleClassfierTrainer import SingleClassfierTrainer
from code.utils.data import get_data_loader_for_classifier, get_orig_d_through_min_max


def do_generate_result_test(data_loader, generator, classifier, label_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = len(label_name)
    result_matrix = np.zeros((n, n))
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x_a, y_a = data
            x_a, y_a = x_a.to(device), y_a.to(device)
            inputs = torch.cat((x_a, y_a), 1)
            outputs = generator(inputs).detach()
            predict = classifier(outputs[:, :16]).detach()
            predicted = torch.argmax(predict, dim=1).cpu()
            labels_cat = torch.argmax(y_a, dim=1).cpu()
            for inx in list(zip(predicted, labels_cat)):
                result_matrix[inx[0], inx[1]] += 1
    return result_matrix


def do_result_test(data_loader, classifier, label_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = len(label_name)
    result_matrix = np.zeros((n, n))
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x_a, y_a = data
            x_a, y_a = x_a.to(device), y_a.to(device)
            inputs = torch.cat((x_a, y_a), 1)
            predict = classifier(inputs[:, :16]).detach()
            predicted = torch.argmax(predict, dim=1).cpu()
            labels_cat = torch.argmax(y_a, dim=1).cpu()
            for inx in list(zip(predicted, labels_cat)):
                result_matrix[inx[0], inx[1]] += 1
    return result_matrix


def get_generated_data(data_loader, generator, columns) -> pd.DataFrame:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = []
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x_a, y_a = data
            x_a, y_a = x_a.to(device), y_a.to(device)
            inputs = torch.cat((x_a, y_a), 1)
            outputs = generator(inputs).detach()[:, :16]
            outputs = torch.cat((outputs, y_a), 1)
            result.append(pd.DataFrame(outputs.cpu().numpy(), columns=columns))
    return pd.concat(result, axis=0)


def get_source_d(sub_dir: str, config: Config) -> pd.DataFrame:
    base = path.join(config.result_dir, sub_dir)
    train_b_fs = path.join(config.result_dir, sub_dir, "dataset_a_train.csv")
    a_train_df = pd.read_csv(train_b_fs)[config.d_16_col + config.selected_column]
    a_train_df = a_train_df[a_train_df[config.validation_feature[0]] == 1]

    val_a_fs = os.path.join(config.result_dir, sub_dir, "dataset_a_val.csv")
    a_val_df = pd.read_csv(val_a_fs)[config.d_16_col + config.selected_column]
    a_val_df = a_val_df[a_val_df[config.validation_feature[0]] == 1]
    return pd.concat([a_train_df, a_val_df], axis=0)


def get_generated_data_d(sub_dir: str, config: Config) -> pd.DataFrame:
    val_a_fs = os.path.join(config.result_dir, sub_dir, "dataset_a_val.csv")
    a_val_df = pd.read_csv(val_a_fs)[config.d_16_col + config.selected_column]
    a_val_df = a_val_df[a_val_df[config.validation_feature[0]] == 1]

    data_loader = get_data_loader_for_classifier([a_val_df], config, False)
    config.model_dir = os.path.join(config.result_dir, sub_dir)
    gan_trainer = GanTrainer5(config,
                              None,
                              None,
                              None)
    return get_generated_data(data_loader, gan_trainer.G_AB, config.d_16_col + config.selected_column)


def get_result_df2(sub_dir: str, config: Config) -> pd.DataFrame:
    base = path.join(config.result_dir, sub_dir)
    train_b_fs = path.join(config.result_dir, sub_dir, "dataset_b_train.csv")
    b_train_df = pd.read_csv(train_b_fs)[config.d_16_col + config.selected_column]

    val_a_fs = os.path.join(config.result_dir, sub_dir, "dataset_a_val.csv")
    a_val_df = pd.read_csv(val_a_fs)[config.d_16_col + config.selected_column]
    a_val_df = a_val_df[a_val_df[config.validation_feature[0]] == 1]

    data_loader = get_data_loader_for_classifier([a_val_df], config, True)
    classifier_b = SingleClassfierTrainer(config, "classifier_b_ground_truth", base, data_loader)
    config.model_dir = os.path.join(config.result_dir, sub_dir)
    gan_trainer = GanTrainer5(config,
                              None,
                              None,
                              None)
    b_val_d = get_generated_data(data_loader, gan_trainer.G_AB, config.d_16_col + config.selected_column)
    tt_df_loader = get_data_loader_for_classifier([b_train_df, b_val_d], config, True)
    result = do_result_test(tt_df_loader, classifier_b.classifier, config.selected_column)
    return pd.DataFrame(result, columns=config.selected_column)


def get_lower_bound(sub_dir: str, config: Config) -> pd.DataFrame:
    base = path.join(config.result_dir, sub_dir)
    val_a_fs = os.path.join(config.result_dir, sub_dir, "dataset_a_val.csv")
    a_val_df = pd.read_csv(val_a_fs)[config.d_16_col + config.selected_column]
    # a_val_df = a_val_df[a_val_df[config.validation_feature[0]] == 1]
    data_loader = get_data_loader_for_classifier([a_val_df], config, True)
    classifier_b = SingleClassfierTrainer(config, "classifier_b_ground_truth", base, data_loader)
    result = do_result_test(data_loader, classifier_b.classifier, config.selected_column)
    return pd.DataFrame(result, columns=config.selected_column)


def get_truth_result(sub_dir, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = path.join(config.result_dir, sub_dir)
    train_b_fs = path.join(config.result_dir, sub_dir, "dataset_b_train.csv")
    b_val_df = pd.read_csv(train_b_fs)[config.d_16_col + config.selected_column]
    orig_d = get_orig_d_through_min_max(config)
    data_loader = get_data_loader_for_classifier([b_val_df, orig_d], config, True)
    classifier_b = SingleClassfierTrainer(config, "classifier_b_ground_truth", base, data_loader)
    classifier = classifier_b.classifier
    n = len(config.selected_column)
    result_matrix = np.zeros((n, n))
    for index, data in enumerate(data_loader):
        x_a, y_a = data
        x_a, y_a = x_a.to(device), y_a.to(device)
        predict = classifier(x_a).detach()
        predicted = torch.argmax(predict, dim=1).cpu()
        labels_cat = torch.argmax(y_a, dim=1).cpu()
        for inx in list(zip(predicted, labels_cat)):
            result_matrix[inx[0], inx[1]] += 1
    return result_matrix


def get_truth_result_5(sub_dir, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = path.join(config.result_dir, sub_dir)
    train_b_fs = path.join(config.result_dir, sub_dir, "dataset_b_val.csv")
    b_val_df = pd.read_csv(train_b_fs)[config.d_16_col + config.selected_column]
    data_loader = get_data_loader_for_classifier([b_val_df], config, True)
    classifier_b = SingleClassfierTrainer(config, "classifier_b_ground_truth", base, data_loader)
    classifier = classifier_b.classifier
    n = len(config.selected_column)
    result_matrix = np.zeros((n, n))
    for index, data in enumerate(data_loader):
        x_a, y_a = data
        x_a, y_a = x_a.to(device), y_a.to(device)
        predict = classifier(x_a).detach()
        predicted = torch.argmax(predict, dim=1).cpu()
        labels_cat = torch.argmax(y_a, dim=1).cpu()
        for inx in list(zip(predicted, labels_cat)):
            result_matrix[inx[0], inx[1]] += 1
    return result_matrix


def get_result_df(sub_dir, config):
    base = path.join(config.result_dir, sub_dir)
    train_a_fs = path.join(config.result_dir, sub_dir, "dataset_a_train.csv")
    a_train_df = pd.read_csv(train_a_fs)[config.d_16_col + config.selected_column]

    val_a_fs = os.path.join(config.result_dir, sub_dir, "dataset_a_val.csv")
    a_val_df = pd.read_csv(val_a_fs)[config.d_16_col + config.selected_column]

    data_loader = get_data_loader_for_classifier([a_val_df], config, True)
    classifier_b = SingleClassfierTrainer(config, "classifier_b_ground_truth", base, data_loader)
    config.model_dir = os.path.join(config.result_dir, sub_dir)
    gan_trainer = GanTrainer5(config,
                              None,
                              None,
                              None)
    result = do_generate_result_test(data_loader, gan_trainer.G_AB, classifier_b.classifier, config.selected_column)
    return pd.DataFrame(result, columns=config.selected_column)
