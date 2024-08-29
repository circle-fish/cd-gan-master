import os
import time

# from code.trainer.GanTrainer3 import DataLoaders, GanTrainer3
from code.trainer.GanTrainer5 import DataLoaders, GanTrainer5
# from code.trainer.GanTrainer4 import GanTrainer4
from code.trainer.SingleClassfierTrainer import SingleClassfierTrainer, SingleClassifierTrainer2
from code.utils import fs, conf, data, test
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from code.utils.conf import get_sum_win, get_spring_win, get_summer_spring
from code.utils.data import get_data_loader_for_classifier
from code.utils.fs import do_common
import numpy as np

from torch.utils.data import DataLoader


def get_largest_number(prefix: str) -> int:
    # experiments = [x for x in os.listdir('./code/training/') if x.startswith(prefix)]
    experiments = [x for x in os.listdir() if x.startswith(prefix)]
    return len(experiments)


def train_gen_2():
    # show current working path
    # print(os.getcwd())
    experiment_prefix = "train_classifier"
    experiment = get_largest_number(experiment_prefix)
    exp_dir = "{}{}".format(experiment_prefix, experiment)
    # fs.make_if_not('./code/training/' + exp_dir)
    fs.make_if_not(exp_dir)
    config_list = [conf.get_sum_win,
                   conf.get_spring_win,
                   conf.get_summer_spring]
    direction_list = ['b_to_a', 'a_to_b']
    print('config_list:', config_list)
    while True:
        for conf_func in config_list:
            for direction in direction_list:
                start_time = time.time()
                config = conf_func(exp_dir, False)
                fs.do_common(config, direction)
                print('config.dir_a:', config.dir_a)
                a_train, a_val, d_train_a, _, _, _ = data.get_full_and_partial_dataset(config, config.dir_a)
                b_train, b_val, d_train_b, val, d_train_c, d_val_c = data.get_full_and_partial_dataset(config,
                                                                                                       config.dir_b)
                b_train_loader = data.get_data_loader_df(b_train, config, True)
                a_train_loader = data.get_data_loader_df(a_train, config, True)
                d_train_loader = data.get_data_loader_df(d_train_a, config, True)
                b_val_loader = data.get_data_loader_df(b_val, config, True)
                a_val_loader = data.get_data_loader_df(a_val, config, True)

                batch_size = config.batch_size
                config.batch_size = 10
                data_loader_b_truth = get_data_loader_for_classifier([b_train, d_train_c], config, True)
                data_loader_b_val = get_data_loader_for_classifier([val, d_val_c], config, True)

                b_full_classifier_trainer = SingleClassifierTrainer2(config, "data_loader_b_truth",
                                                                     config.classifier_dir, data_loader_b_truth,
                                                                     data_loader_b_val)
                b_full_classifier_trainer.train()
                b_full_classifier_trainer.save_model()

                config.batch_size = batch_size
                data_loader = DataLoaders(b_train_loader, a_train_loader, d_train_loader, b_val_loader, a_val_loader)
                trainer = GanTrainer5(config, data_loader, test.do_classifier_test, test.do_generate_result_test,
                                      b_full_classifier_trainer.classifier)
                trainer.train()
                end_time = time.time()
                print("bidirectional training take {} minuts".format(
                    (end_time - start_time) / 60))


def train_classifier_directly():
    experiment_prefix = "train_classifier"
    experiment = get_largest_number(experiment_prefix)
    exp_dir = "{}{}".format(experiment_prefix, experiment)
    fs.make_if_not(exp_dir)
    config_list = [conf.get_sum_win,
                   conf.get_spring_win, conf.get_summer_spring]
    direction_list = ['a_to_b', 'b_to_a']

    while True:
        for conf_func in config_list:
            for direction in direction_list:
                start_time = time.time()
                config = conf_func(exp_dir)
                fs.do_common(config, direction)
                a_train, a_val, d_train_a = data.get_full_and_partial_dataset(config, config.dir_a)
                b_train, b_val, d_train_b = data.get_full_and_partial_dataset(config, config.dir_b)

                b_train_loader = data.get_data_loader_df(b_train, config, True)
                a_train_loader = data.get_data_loader_df(a_train, config, True)
                d_train_loader = data.get_data_loader_df(d_train_a, config, True)
                b_val_loader = data.get_data_loader_df(b_val, config, True)
                a_val_loader = data.get_data_loader_df(a_val, config, True)

                data_loader = DataLoaders(b_train_loader, a_train_loader, d_train_loader, b_val_loader, a_val_loader)
                trainer = GanTrainer5(config, data_loader, test.do_classifier_test)
                trainer.train()
                end_time = time.time()
                print("bidirectional training take {} minuts".format((end_time - start_time) / 60))


def train_classifier_with_discriminator():
    experiment_prefix = "train_c_with_d"
    experiment = get_largest_number(experiment_prefix)
    exp_dir = "{}{}".format(experiment_prefix, experiment)
    fs.make_if_not(exp_dir)
    config_list = [conf.get_sum_win,
                   conf.get_spring_win, conf.get_summer_spring]
    direction_list = ['a_to_b', 'b_to_a']

    while True:
        for conf_func in config_list:
            for direction in direction_list:
                start_time = time.time()
                config = conf_func(exp_dir)
                fs.do_common(config, direction)
                a_train, a_val, d_train_a = data.get_full_and_partial_dataset(
                    config, config.dir_a)
                b_train, b_val, d_train_b = data.get_full_and_partial_dataset(
                    config, config.dir_b)
                b_train_loader = data.get_data_loader_df(b_train, config, True)
                a_train_loader = data.get_data_loader_df(a_train, config, True)
                d_train_loader = data.get_data_loader_df(
                    d_train_a, config, True)
                b_val_loader = data.get_data_loader_df(b_val, config, True)
                a_val_loader = data.get_data_loader_df(a_val, config, True)

                data_loader = DataLoaders(b_train_loader, a_train_loader, d_train_loader, b_val_loader, a_val_loader)
                trainer = GanTrainer5(config, data_loader, test.do_classifier_test)
                trainer.train()
                end_time = time.time()
                print("bidirectional training take {} minuts".format((end_time - start_time) / 60))


def train_classifier_with_d_all_col():
    experiment_prefix = "train_c_with_d_all_col"
    experiment = get_largest_number(experiment_prefix)
    exp_dir = "{}{}".format(experiment_prefix, experiment)
    fs.make_if_not(exp_dir)
    config_list = [conf.get_sum_win,
                   conf.get_spring_win, conf.get_summer_spring]
    direction_list = ['a_to_b', 'b_to_a']

    while True:
        for conf_func in config_list:
            for direction in direction_list:
                start_time = time.time()
                config = conf_func(exp_dir)
                fs.do_common(config, direction)
                config.d_16_col = [
                    'SYS-CTL', 'RF-CTRL', 'ECONCTRL', 'ACCH-SEL', 'HWC-VLV', 'CHWC-VLV', 'EA-DMPR', 'RA-DMPR',
                    'OA-DMPR', 'SF-SST', 'RF-SST', 'SF-WAT', 'RF-WAT', 'HWP-GPM', 'CHWP-GPM', 'RFCFMLAG', 'RF-SFSPD',
                    'SAT_SPT', 'SA_SPSPT', 'RMT-CFM', 'SA-CFM', 'RA-CFM', 'OA-CFM', 'SA-TEMP', 'MA-TEMP', 'RA-TEMP',
                    'HWC-DAT', 'CHWC-DAT', 'SA-SP', 'SF-DP', 'RF-DP', 'SF-SPD', 'RF-SPD', 'SA-HUMD', 'RA-HUMD',
                    'OA-TEMP', 'OAD-TEMP', 'HWC-EWT', 'HWC-LWT', 'HWC-MWT', 'CHWC-EWT', 'CHWC-LWT', 'CHWC-MWT',
                    'HWP-DP', 'CHWC-EAH', 'CHWC-LAH', 'RM-TEMP', 'PLN-TEMP', 'VAV-EAT', 'VAV-DAT', 'VAVHCVLV',
                    'VAV-DMPR', 'RMCLGSPT', 'RMHTGSPT', 'OCC-MAX', 'OCC-MIN', 'VAV-DP', 'VAVCFMDP', 'VAVHCEWT',
                    'VAVHCLWT', 'VAVHCGPM', 'RM-TEMP.1', 'PLN-TEMP.1', 'VAV-EAT.1', 'VAV-DAT.1', 'VAVHCVLV.1',
                    'VAV-DMPR.1', 'RMCLGSPT.1', 'RMHTGSPT.1', 'OCC-MAX.1', 'OCC-MIN.1', 'VAV-DP.1', 'VAVCFMDP.1',
                    'VAVHCEWT.1', 'VAVHCLWT.1', 'VAVHCGPM.1', 'RM-TEMP.2', 'PLN-TEMP.2', 'VAV-EAT.2', 'VAV-DAT.2',
                    'VAVHCVLV.2', 'VAV-DMPR.2', 'RMCLGSPT.2', 'RMHTGSPT.2', 'OCC-MAX.2', 'OCC-MIN.2', 'VAV-DP.2',
                    'VAVCFMDP.2', 'VAVHCEWT.2', 'VAVHCLWT.2', 'VAVHCGPM.2', 'RM-TEMP.3', 'PLN-TEMP.3', 'VAV-EAT.3',
                    'VAV-DAT.3', 'VAVHCVLV.3', 'VAV-DMPR.3', 'RMCLGSPT.3', 'RMHTGSPT.3', 'OCC-MAX.3', 'OCC-MIN.3',
                    'VAV-DP.3', 'VAVCFMDP.3', 'VAVHCEWT.3', 'VAVHCLWT.3', 'VAVHCGPM.3', 'IABBHT1', 'IABBHT2', 'IBBBHT1',
                    'IBBBHT2', 'WABBHT1', 'WABBHT2', 'WBBBHT1', 'WBBBHT2', 'SABBHT1', 'SABBHT2', 'SBBBHT1', 'SBBBHT2',
                    'EABBHT1', 'EABBHT2', 'EBBBHT1', 'EBBBHT2', 'IBBAMPS', 'WBBAMPS', 'SBBAMPS', 'EBBAMPS', 'IALITES1',
                    'IALITES2', 'IBLITES1', 'IBLITES2', 'WALITES1', 'WALITES2', 'WBLITES1', 'WBLITES2', 'SALITES1',
                    'SALITES2', 'SBLITES1', 'SBLITES2', 'EALITES1', 'EALITES2', 'EBLITES1', 'EBLITES2', 'IALITWAT',
                    'WALITWAT', 'SALITWAT', 'EALITWAT', 'IBLITWAT', 'WBLITWAT', 'SBLITWAT', 'EBLITWAT',
                    'CHWC_GPM', 'E_hcoil', 'E_ccoil', 'E_SF', 'E_RF', 'E_ZONE_I', 'E_ZONE_W', 'E_ZONE_S', 'E_ZONE_E'
                ]
                # 'HWC_GPM'
                a_train, a_val, d_train_a = data.get_full_and_partial_dataset(
                    config, config.dir_a)
                b_train, b_val, d_train_b = data.get_full_and_partial_dataset(
                    config, config.dir_b)
                b_train_loader = data.get_data_loader_df(b_train, config, True)
                a_train_loader = data.get_data_loader_df(a_train, config, True)
                d_train_loader = data.get_data_loader_df(d_train_a, config, True)
                b_val_loader = data.get_data_loader_df(b_val, config, True)
                a_val_loader = data.get_data_loader_df(a_val, config, True)

                data_loader = DataLoaders(b_train_loader, a_train_loader, d_train_loader, b_val_loader, a_val_loader)
                trainer = GanTrainer5(config, data_loader, test.do_classifier_test)
                trainer.train()
                end_time = time.time()
                print("bidirectional training take {} minuts".format((end_time - start_time) / 60))


def knn_result():
    prefix = 'knn1'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, False), get_spring_win(
        prefix, False), get_sum_win(prefix, False)]

    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            b_train, b_val, d_train_b = data.get_full_and_partial_dataset(config, config.dir_b)
            train = pd.concat([b_train, d_train_b], axis=0)
            n = len(config.selected_column)
            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(train[config.d_16_col], train[config.selected_column])
            test_x, test_y = b_val[config.d_16_col].values, b_val[config.selected_column].values
            result = neigh.predict(test_x)

            result_matrix = np.zeros((n, n))
            for i in range(len(b_val)):
                target = np.argmax(result[i])
                source = np.argmax(test_y[i])
                result_matrix[target, source] += 1
            df = pd.DataFrame(result_matrix)
            df.to_csv("{}-{}".format(config.super_base, direction), index=False, header=False)
