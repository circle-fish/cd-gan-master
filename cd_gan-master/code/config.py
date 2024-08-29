import os.path as path

import pandas as pd
import torch


class Config(object):
    def __init__(self, super_base):
        self.super_base = super_base
        # Data Loaders (Filled by Main)
        self.data_loader_a_val = None
        self.data_loader_b_val = None
        self.data_loader_a_train = None
        self.data_loader_b_train = None
        self.data_loader_a_test = None
        self.data_loader_b_test = None

        self.data_loader_b_clas_train = None
        self.data_loader_b_clas_test = None
        self.data_set_b = None
        self.data_set_a = None
        # Data
        self.base_dir = "./code/trainer/data/"
        self.model_dir = path.join(self.super_base, 'model')
        self.classifier_dir = path.join(self.model_dir, 'model')
        self.dir_list = ["2007_summer", '2008_spring', '2008_winter']
        self.dir_a = None
        self.dir_b = None

        # Training Params
        self.num_gpu = 1
        self.learning_rate = 0.0002
        self.loss = 'log_prob'
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.optimizer = 'adam'
        self.weight_decay = 0.0001
        self.cnn_type = 0
        self.amsgrad = True

        # For classifier
        self.epoch = 30
        self.classifier_name = 'Classifier_A'

        # For generation
        self.gen_time = 40

        # Networks
        self.num_workers = 2
        self.load_path = False
        self.start_step = 0
        self.log_step = 200
        # self.max_step = 30000
        self.max_step = 3000
        self.save_step = 200
        self.g_num_layer = 5
        self.d_num_layer = 5
        self.use_cnn = False

        self.batch_size = 10
        self.fc_hidden_dim = 256

        # Training ratio
        self.ratio_d_thresh = 45
        self.ratio_in_total = 50
        self.ratio_c_d_thresh = 5
        self.ratio_d_d_thresh = 5

        # self.dir_list = ["2007_summer", '2008_spring', '2008_winter']
        # self.dir_a = './code/trainer/data/2007_summer/'
        # self.dir_b = './code/trainer/data/2008_spring/'
        # Summer <-> Spring
        # NORMAL
        # CCVS_FC, CCVS_FO
        # OADS_FC

        # Spring <-> Winter
        # NORMAL
        # CCVS_FO, EADS_FO
        # EADS_FC

        # Summer <-> Winter
        # NORMAL
        # CCVS_FO, EADS_FC
        # EADS_FO

        # Faults to focus on
        self.normal = ['NORMAL']
        self.common_feature = ['CCVS_FC', 'CCVS_FO']  # 'EADS_FO',
        self.validation_feature = ['OADS_FC']  # EADS_FC
        # self.tranferred_feature = ['CCVS_O15']  # CCVS_O65, CCV_CU
        self.selected_column = self.normal + self.common_feature + self.validation_feature
        self.d_16_col = [
            'HWC-VLV',
            'E_hcoil',
            'CHWC-VLV',
            'E_ccoil',
            'SF-SPD',
            'E_SF',
            'RF-SPD',
            'E_RF',
            'SA-CFM',
            'RA-CFM',
            'OA-CFM',
            'SA-TEMP',
            'MA-TEMP',
            'RA-TEMP',
            'HWC-DAT',
            'CHWC-DAT']

        self.error_cat = pd.api.types.CategoricalDtype(
            categories=self.selected_column, ordered=False)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def get_pd_data(self, folder, errors):
        dataset = []
        dir_root = path.join(self.base_dir, folder)
        for error_file in errors:
            print('errors:', error_file)
            files = [path.join(dir_root, error_file + '_A.csv')]
            print('files:', files)
            for file in files:
                if path.exists(file):
                    dataset.append(pd.read_csv(file))
        dataset = pd.concat(dataset)
        dataset = dataset[['FAULT'] + self.d_16_col]
        dataset['y'] = dataset['FAULT'].astype(self.error_cat)
        y_default = dataset['FAULT']
        dataset.drop(['FAULT'], axis=1, inplace=True)
        y_a = pd.get_dummies(dataset['y'], columns=['y'])
        dataset.drop(['y'], axis=1, inplace=True)
        return dataset, y_a, y_default
