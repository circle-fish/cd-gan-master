import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from typing import Union
from code.config import Config
from code.model.AhuSimpleDataset import AhuSimpleDataset

Data = Union[np.array, pd.DataFrame]


def get_data_loader_from_file(full_path, config: Config, shuffle: bool) -> DataLoader:
    df = pd.read_csv(full_path)
    df_x = df[config.d_16_col]
    df_y = df[config.selected_column]
    return get_data_loader(df_x, df_y, config, shuffle)


def save(data_set, full_path):
    df = pd.concat([data_set.x, data_set.y], axis=1)
    df.to_csv(full_path)


def get_data_loader(df_x: Data, df_y: Data, config: Config, shuffle: bool) -> DataLoader:
    d_16_map = {x: np.float32 for x in config.d_16_col}
    df_x = df_x.astype(d_16_map, copy=False)
    d_16_map = {x: np.float32 for x in config.selected_column}
    df_y = df_y.astype(d_16_map, copy=False)
    data_set = AhuSimpleDataset(df_x, df_y)
    #     print(df_x.shape)
    loader = DataLoader(data_set, batch_size=config.batch_size,
                        shuffle=shuffle, num_workers=config.num_workers)
    return loader


def get_data_loader_df(df: pd.DataFrame, config: Config, shuffle: bool) -> DataLoader:
    df_x, df_y = df[config.d_16_col], df[config.selected_column]
    d_16_map = {x: np.float32 for x in config.d_16_col}
    df_x = df_x.astype(d_16_map, copy=False)
    d_16_map = {x: np.float32 for x in config.selected_column}
    df_y = df_y.astype(d_16_map, copy=False)
    data_set = AhuSimpleDataset(df_x, df_y)
    #     print(df_x.shape)
    loader = DataLoader(data_set, batch_size=config.batch_size,
                        shuffle=shuffle, num_workers=config.num_workers)
    return loader

def get_data_loader_season(df: pd.DataFrame, config: Config, shuffle: bool) -> DataLoader:
    df_x, df_y = df[config.d_16_col], df[config.selected_column]
    d_16_map = {x: np.float32 for x in config.d_16_col}
    df_x = df_x.astype(d_16_map, copy=False)
    d_16_map = {x: np.float32 for x in config.selected_column}
    df_y = df_y.astype(d_16_map, copy=False)
    data_set = AhuSimpleDataset(df_x, df_y)
    #     print(df_x.shape)
    loader = DataLoader(data_set, batch_size=config.batch_size,
                        shuffle=shuffle, num_workers=config.num_workers)
    return loader


def k_fold_split(dataset, n_splits, shuffle, config):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    x, y, z = dataset[config.d_16_col], dataset.drop(config.d_16_col + ["FAULT"], axis=1), dataset['FAULT']

    index = skf.split(x, z)
    train_index_list = []
    test_index_list = []
    for t_inx, s_inx in index:
        train_index_list.append(t_inx)
        test_index_list.append(s_inx)

    picked_index = random.randrange(len(train_index_list))
    train_index = train_index_list[picked_index]
    test_index = test_index_list[picked_index]
    print("len(train)", len(t_inx), "len(test)", len(s_inx))
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    return x_train, y_train, x_test, y_test


def get_full_and_partial_dataset(config, directory):
    print("Loading {}".format(directory))
    df_a_x, df_a_y, df_a_y_default = config.get_pd_data(directory, config.selected_column)
    df_a = pd.concat([df_a_x, df_a_y, df_a_y_default], axis=1)
    normals = df_a[df_a['NORMAL'] == 1]
    droped = df_a[df_a['NORMAL'] != 1]
    sampled_normal = normals
    # .sample(2880)
    df_a = pd.concat([sampled_normal, droped], axis=0)
    prepro = MinMaxScaler()
    df_a.fillna(0, inplace=True)

    # null_columns = df_a.columns[df_a.isnull().any()]
    # df_a[null_columns].isnull().sum()
    # print(df_a[df_a.isnull().any(axis=1)].head())

    df_a.loc[:, config.d_16_col] = prepro.fit_transform(df_a[config.d_16_col])
    d_16_map = {x: np.float32 for x in config.d_16_col + config.selected_column}
    df_a = df_a.astype(d_16_map, copy=False)
    df_a = df_a.sort_values(config.selected_column, axis=0)
    test_dataset = []
    for col in config.validation_feature:
        test_dataset.append(df_a[df_a[col] == 1])
        df_a = df_a[df_a[col] == 0]
    validation_df = pd.concat(test_dataset, axis=0)
    d_train, d_val = train_test_split(validation_df, test_size=0.5)
    d_train_c, d_val_c = train_test_split(validation_df, test_size=0.8)
        #= validation_df.sample(720)


    # df_a = (a,b,c)
    train_a_partial_df = df_a
    train_x, train_y, test_x, test_y = k_fold_split(train_a_partial_df, 9, True, config)
    train_df = pd.concat([train_x, train_y], axis=1)
    validation_df_final = pd.concat([test_x, test_y], axis=1)
    val_df = pd.concat([validation_df_final, d_val], axis=0, sort=False)
    return train_df, val_df, d_train, validation_df_final, d_train_c, d_val_c


def get_data_loader_for_classifier(data_list, config, shuffle):
    df = pd.concat(data_list, axis=0)
    df_x = df[config.d_16_col]
    df_y = df[config.selected_column]
    return get_data_loader(df_x, df_y, config, shuffle)


def get_data_loader_for_gan(data_list_a, data_list_b, config):
    df = pd.concat(data_list_a, axis=0)
    df_x = df[config.d_16_col]
    df_y = df[config.selected_column]
    if len(data_list_b) == 0:
        df_x_b = []
        df_y_b = []
    else:
        df = pd.concat(data_list_b, axis=0)
        df_x_b = df[config.d_16_col]
        df_y_b = df[config.selected_column]
    return get_data_loader(df_x, df_y, df_x_b, df_y_b)


def get_orig_d_through_min_max(config: Config):
    df_a_x, df_a_y, df_a_y_default = config.get_pd_data(config.dir_b, config.selected_column)
    df_a = pd.concat([df_a_x, df_a_y, df_a_y_default], axis=1)
    normals = df_a[df_a['NORMAL'] == 1]
    droped = df_a[df_a['NORMAL'] != 1]
    sampled_normal = normals
    # .sample(2880)
    df_a = pd.concat([sampled_normal, droped], axis=0)
    prepro = MinMaxScaler()
    df_a.fillna(0, inplace=True)
    df_a.loc[:, config.d_16_col] = prepro.fit_transform(df_a[config.d_16_col])
    d_16_map = {x: np.float32 for x in config.d_16_col + config.selected_column}
    df_a = df_a.astype(d_16_map, copy=False)
    df_a = df_a.sort_values(config.selected_column, axis=0)
    test_dataset = []
    for col in config.validation_feature:
        test_dataset.append(df_a[df_a[col] == 1])
        df_a = df_a[df_a[col] == 0]
    validation_df = pd.concat(test_dataset, axis=0)
    return validation_df
