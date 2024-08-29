from code.config import Config
import pandas as pd


def get_sum_win(prefix: str, load_path: bool) -> Config:
    config = Config('{}/sum_win'.format(prefix))
    config.common_feature = ['EADS_FC', 'CCVS_FO']  # 'EADS_FO',
    config.validation_feature = ['EADS_FO']  # EADS_FC
    config.selected_column = config.common_feature + config.validation_feature + config.normal
    config.error_cat = pd.api.types.CategoricalDtype(categories=config.selected_column, ordered=False)
    # config.dir_a = config.dir_list[0]
    # config.dir_b = config.dir_list[2]
    config.dir_a = '2007_summer'
    config.dir_b = '2008_winter'
    config.load_path = load_path
    return config


def get_spring_win(prefix: str, load_path: bool) -> Config:
    config = Config('{}/spring_win'.format(prefix))
    config.common_feature = ['EADS_FO', 'CCVS_FO']  # 'EADS_FO',
    config.validation_feature = ['EADS_FC']  # EADS_FC
    config.selected_column = config.common_feature + config.validation_feature + config.normal
    config.error_cat = pd.api.types.CategoricalDtype(categories=config.selected_column, ordered=False)
    # config.dir_a = config.dir_list[1]
    # config.dir_b = config.dir_list[2]
    config.dir_a = '2008_spring'
    config.dir_b = '2008_winter'
    config.load_path = load_path
    return config


def get_summer_spring(prefix: str, load_path: bool) -> Config:
    config = Config('{}/sum_spring'.format(prefix))
    config.common_feature = ['CCVS_FC', 'CCVS_FO']  # 'EADS_FO',
    config.validation_feature = ['OADS_FC']  # EADS_FC
    # config.tranferred_feature = ['CCVS_O15']  # CCVS_O65, CCV_CU
    config.selected_column = config.common_feature + config.validation_feature + config.normal
    config.error_cat = pd.api.types.CategoricalDtype(categories=config.selected_column, ordered=False)
    # config.dir_a = config.dir_list[0]
    # config.dir_b = config.dir_list[1]
    config.dir_a = '2007_summer'
    config.dir_b = '2008_spring'
    config.load_path = load_path
    return config
