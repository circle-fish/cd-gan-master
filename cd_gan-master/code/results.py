import os
from os import path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import statsmodels.api as sm

from code.trainer.GanTrainer2 import GanTrainer2
from code.utils import data
from code.utils.conf import get_summer_spring, get_spring_win, get_sum_win
from code.utils.data import get_orig_d_through_min_max, get_data_loader_for_classifier
from code.utils.fs import do_common
from code.utils.image import get_matrix, show_and_save_image
from code.utils.result import get_result_df, get_result_df2, get_generated_data_d, get_truth_result, get_generated_data, \
    get_lower_bound, get_truth_result_5, get_source_d


def get_backup_3_truth_b():
    matrix = np.array([[0.498, 0.031, 0.170, 0.038],
                       [0.048, 0.419, 0.033, 0.123],
                       [0.390, 0.446, 0.658, 0.467],
                       [0.064, 0.103, 0.138, 0.380]])
    show_and_save_image(matrix, "{}-{}-{}.png".format('spring_win', 'b_to_a', 'only_d-a'),
                        ['EADS_FO', 'CCVS_FO', 'EADS_FC', 'NORMAL'])


def get_backup_3_truth_a():
    matrix = np.array([[0.342, 0.002, 0.134, 0.018],
                       [0.122, 0.943, 0.345, 0.022],
                       [0.015, 0.006, 0.482, 0.042],
                       [0.522, 0.049, 0.038, 0.917]])
    show_and_save_image(matrix, "{}-{}-{}.png".format('spring_win', 'b_to_a', 'truth_a'),
                        ['EADS_FO', 'CCVS_FO', 'EADS_FC', 'NORMAL'])

    matrix = np.array([[0.568, 0.017, 0.003, 0.060],
                       [0.042, 0.929, 0.000, 0.075],
                       [0.011, 0.000, 0.677, 0.034],
                       [0.379, 0.054, 0.321, 0.83]])
    show_and_save_image(matrix, "{}-{}-{}.png".format('sum_win', 'b_to_a', 'truth_a'),
                        ['EADS_FC', 'CCVS_FO', 'EADS_FO', 'NORMAL'])


def get_backup_3_truth():
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]

    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            print('working on [{}]'.format(config.result_dir))
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            result_list = []
            for sub_dir in result_folders:
                print('Get df result d from [{}]'.format(sub_dir))
                result_list.append(get_truth_result(sub_dir, config))
            matrix = get_matrix(result_list)
            result = np.average(matrix, axis=0)
            sum_s = np.sum(result, axis=0)
            res_mt = result / sum_s
            show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'truth'),
                                config.selected_column)


def get_backup_5_truth():
    prefix = 'backup5'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            if config == config_list[0] and direction == 'b_to_a':
                continue
            do_common(config, direction)
            print('working on [{}]'.format(config.result_dir))
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            result_list = []
            for sub_dir in result_folders:
                print('Get df result d from [{}]'.format(sub_dir))
                result_list.append(get_truth_result_5(sub_dir, config))
            matrix = get_matrix(result_list)
            result = np.average(matrix, axis=0)
            sum_s = np.sum(result, axis=0)
            res_mt = result / sum_s
            show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'truth'),
                                config.selected_column)


def get_backup_4_truth():
    prefix = 'backup4'
    direction_list = ['b_to_a']
    config_list = [get_summer_spring(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            # if config == config_list[0] and direction == 'b_to_a':
            #     continue
            do_common(config, direction)
            print('working on [{}]'.format(config.result_dir))
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            result_list = []
            for sub_dir in result_folders:
                print('Get df result d from [{}]'.format(sub_dir))
                result_list.append(get_truth_result(sub_dir, config))
            matrix = get_matrix(result_list)
            result = np.average(matrix, axis=0)
            sum_s = np.sum(result, axis=0)
            res_mt = result / sum_s
            show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'truth'),
                                config.selected_column)


def get_backup_3_results():
    prefix = 'backup3'
    direction_list = ['b_to_a']
    # , get_spring_win(prefix, True), get_sum_win(prefix, True)]
    config_list = [get_summer_spring(prefix, True)]

    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            print('working on [{}]'.format(config.result_dir))
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]

            result_list = []
            for sub_dir in result_folders:
                print('Get df result d from [{}]'.format(sub_dir))
                result_list.append(get_result_df2(sub_dir, config))
            matrix = get_matrix(result_list)

            result = np.average(matrix, axis=0)
            sum_s = np.sum(result, axis=0)
            res_mt = result / sum_s
            show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'only_d'),
                                config.selected_column)

            result_list = []
            for sub_dir in result_folders:
                print('Get df result from [{}]'.format(sub_dir))
                result_list.append(get_result_df(sub_dir, config))
            matrix = get_matrix(result_list)

            result = np.average(matrix, axis=0)
            sum_s = np.sum(result, axis=0)
            res_mt = result / sum_s
            show_and_save_image(
                res_mt, "{}-{}.png".format(config.super_base, direction), config.selected_column)


def get_backup_3_results_high_low():
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    # ['sum_to_sp', 'sp_to_sum', 'sum_to_win', 'win_to_sum', 'sp_to_win', 'win_to_sp']
    config_list = [get_summer_spring(prefix, True), get_sum_win(
        prefix, True), get_spring_win(prefix, True)]
    high = []
    low = []
    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            print('working on [{}]'.format(config.result_dir))
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]

            result_list = []
            for sub_dir in result_folders:
                print('Get df result d from [{}]'.format(sub_dir))
                result_list.append(get_result_df2(sub_dir, config))
            matrix = get_matrix(result_list)

            mama = matrix[:, 2, 2] / 1440
            result = np.percentile(mama, q=30)
            low.append(result * 100)
            # sum_s = np.sum(result, axis=0)
            # res_mt = result / sum_s
            # show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'lower_spread'),
            #                     config.selected_column)

            hi = np.percentile(mama, q=70)
            high.append(hi * 100)
            # result = np.percentile(matrix, axis=0, q=90)
            # sum_s = np.sum(result, axis=0)
            # res_mt = result / sum_s
            # show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'higher_spread'),
            #                     config.selected_column)
    print(low)
    print(high)


def qq_plot():
    '''
    read all data, apply preprocessing, get label d, get generated data.
    :return:
    '''
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            orig_d = get_orig_d_through_min_max(config)[config.d_16_col]
            y_2 = range(1, len(orig_d) + 1)
            y_2 = sm.add_constant(y_2)
            model_2 = sm.OLS(y_2, orig_d).fit()
            res = model_2.resid
            fig = sm.qqplot(res, line='s')
            # plt.show()
            fig.savefig("qq/{}-{}-q-q-origin-d.png".format(config.super_base.split('/')[-1], direction),
                        bbox_inches='tight')

            generated = []
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            for sub_dir in result_folders:
                generated_d = get_generated_data_d(sub_dir, config)
                generated.append(generated_d)
            combined = pd.concat(generated, axis=0)
            y_1 = range(1, len(combined) + 1)
            y_1 = sm.add_constant(y_1)
            model_1 = sm.OLS(y_1, combined)
            mod_fit = model_1.fit()
            res = mod_fit.resid
            fig = sm.qqplot(res, line='s')
            # plt.show()
            fig.savefig("qq/{}-{}-q-q-generated-d.png".format(config.super_base.split('/')[-1], direction),
                        bbox_inches='tight')


def t_sne_result_source():
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            orig_d = get_orig_d_through_min_max(config)[config.d_16_col]
            X_embedded = TSNE(n_components=2).fit_transform(orig_d)
            true_x, true_y = zip(*X_embedded)
            # true_c = ['g' for i in range(len(true_x))]
            generated = []
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            for i in range(len(result_folders)):
                sub_dir = result_folders[i]
                generated_d = get_source_d(sub_dir, config)
                combined_d = TSNE(n_components=2).fit_transform(generated_d)
                x, y = zip(*combined_d)
                fig, ax = plt.subplots()
                ax.scatter(true_x, true_y, c='g', marker="o",
                           label="ground truth", alpha=0.3)
                # fig.savefig("{}-tnse-{}-true".format(config.super_base, direction))
                ax.scatter(x, y, c='b', marker='s',
                           label="generation", alpha=0.3)
                # ax.legend()
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                fig.savefig("{}/{}-tnse-{}-together-{}-source".format('tsne2', config.super_base.split('/')[-1], direction, i),
                            bbox_inches='tight')
                generated.append(generated_d)


def t_nse_result():
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            orig_d = get_orig_d_through_min_max(config)[config.d_16_col]
            X_embedded = TSNE(n_components=2).fit_transform(orig_d)
            true_x, true_y = zip(*X_embedded)
            # true_c = ['g' for i in range(len(true_x))]
            generated = []
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            for i in range(len(result_folders)):
                sub_dir = result_folders[i]
                generated_d = get_generated_data_d(sub_dir, config)
                combined_d = TSNE(n_components=2).fit_transform(generated_d)
                x, y = zip(*combined_d)
                fig, ax = plt.subplots()
                ax.scatter(true_x, true_y, c='g', marker="o",
                           label="ground truth", alpha=0.3)
                # fig.savefig("{}-tnse-{}-true".format(config.super_base, direction))
                ax.scatter(x, y, c='b', marker='s',
                           label="generation", alpha=0.3)
                # ax.legend()
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                fig.savefig("{}/{}-tnse-{}-together-{}".format('tsne1', config.super_base.split('/')[-1], direction, i),
                            bbox_inches='tight')
                generated.append(generated_d)


def make_summary():
    # figsize = (15, 5)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots(15, 5)
    # print(ax)
    width = 0.35
    plt.grid(True)
    y = [62.9, 65.8, 49.2, 46.7, 81.0, 60.3]

    # y_u = [36.54861111111111, 49.333333333333335, 28.944444444444445, 29.72222222222223, 37.611111111111106,
    #        46.805555555555556]
    # y_l = [100.0, 100.0, 100.0, 51.21527777777778, 100.0, 100]

    # y_u = [68.8, 66, 50, 47, 82, 61]
    # y_l = [27.4, 63, 45, 43, 77, 56]
    # y_u = [49.583333333333336, 49.65277777777778, 35.0, 45.52777777777778, 50.06944444444444, 48.05555555555556]
    # y_l = [73.22222222222219, 94.83333333333333, 49.65277777777778, 48.736111111111114, 100.0, 50.34722222222222]

    y_up = [79.8, 68.1, 79.7, 44.8, 84.6, 93.5]
    y_low = [32.6, 42.1, 4.4, 0, 73.3, 19.4]

    x = np.arange(1, 7) * 10
    # [1, 2, 3, 4, 5, 6]

    bar1 = ax.bar(x - 8, y_up, width=2, color='blue',
                  edgecolor='white', label='On Target Domain')
    bar2 = ax.bar(x - 6, y, width=2, color="orange",
                  edgecolor='white', label='On Generated Data')
    bar3 = ax.bar(x - 4, y_low, width=2, color='green',
                  edgecolor='white', label='On Source Domain')
    # plt.fill_between(x, y_low, y_up, color='blue', alpha=0.1)
    # plt.fill_between(x, y_l, y_u, color='blue', alpha=0.3)
    ax.set(xticks=x - 6,
           # yticks=np.arange(len(labels)),
           # ... and label them with the respective list entries
           xticklabels=['sum_to_sp', 'sp_to_sum', 'sum_to_win',
                        'win_to_sum', 'sp_to_win', 'win_to_sp'],
           # yticklabels=labels,
           xlabel='Different Transformation Cases',
           ylabel='Accuracy Values (%)')

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    "{:0.1f}".format(height),
                    ha='center', va='bottom')

    autolabel(bar1)
    autolabel(bar2)
    autolabel(bar3)

    # plt.ylabel('Accuracy Values (%)')
    # plt.xlabel('Different Transformation Cases')
    # plt.xticks(x - 6, ('sum_to_sp', 'sp_to_sum', 'sum_to_win', 'win_to_sum', 'sp_to_win', 'win_to_sp'))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # fig.tight_layout()
    fig.savefig('accuracy_summary.png', bbox_inches='tight', pad_inches=0)


def make_summary_direction():
    fig = plt.figure(figsize=(15, 5))
    plt.grid(True)
    # ('sum_to_sp', 'sp_to_sum', 'sum_to_win', 'win_to_sum', 'sp_to_win', 'win_to_sp')
    y_list = [
        [18.9, 71.0, 62.9, 98.6],
        [16.6, 69.3, 65.8, 85.3],
        [17.6, 87.5, 49.2, 90.7],
        [59.9, 91.2, 46.7, 52.6],
        [18.6, 43.7, 81.0, 41.2],
        [6.30, 81.7, 60.3, 99.9]
    ]
    # 62.9, 65.8, 49.2, 46.7, 81.0, 60.3
    # y_u = [36.54861111111111, 49.333333333333335, 28.944444444444445, 29.72222222222223, 37.611111111111106,
    #        46.805555555555556]
    # y_l = [100.0, 100.0, 100.0, 51.21527777777778, 100.0, 100]

    # y_u = [68.8, 66, 50, 47, 82, 61]
    # y_l = [27.4, 63, 45, 43, 77, 56]
    # y_u = [49.583333333333336, 49.65277777777778, 35.0, 45.52777777777778, 50.06944444444444, 48.05555555555556]
    # y_l = [73.22222222222219, 94.83333333333333, 49.65277777777778, 48.736111111111114, 100.0, 50.34722222222222]

    y_up_list = [
        [94.5, 98.8, 79.8, 99.6],
        [74.5, 84.4, 52.8, 90.6],
        [78.8, 90.9, 79.7, 96.7],
        [59.3, 92.3, 44.8, 85.3],
        [70.3, 93.1, 84.6, 94.2],
        [67.0, 97.2, 93.5, 87.1]
    ]
    # 79.8, 68.1, 79.7, 44.8, 84.6, 93.5
    y_low_list = [
        [50.6, 75.9, 32.6, 62.7],
        [20.2, 84.4, 42.1, 45.9],
        [7.90, 47.0, 4.40, 45.9],
        [0.10, 99.2, 0.00, 2.60],
        [11.0, 33.1, 73.3, 29.4],
        [35.6, 65.0, 19.4, 45.7]
    ]
    # [32.6, 42.1, 4.4, 0, 73.3, 19.4]

    labels = ['sum_to_sp', 'sp_to_sum', 'sum_to_win',
              'win_to_sum', 'sp_to_win', 'win_to_sp']
    xlabels = ['Summer to Spring', 'Spring to Summer', 'Summer to Winter', 'Winter to Summer', 'Spring to Winter',
               'Winter to Spring']
    x = [1, 2, 3, 4]
    for i in range(len(y_low_list)):
        y = y_list[i]
        y_up = y_up_list[i]
        y_low = y_low_list[i]
        fig, ax = plt.subplots()
        plt.bar(x, y, '*-', label='Transferred Data')
        plt.bar(x, y_up, 'o-', label='Upper Bound')
        plt.bar(x, y_low, '^-', label='Lower Bound')
        plt.fill_between(x, y_low, y_up, color='blue', alpha=0.1)
        # plt.fill_between(x, y_l, y_u, color='blue', alpha=0.3)
        plt.legend()
        plt.ylabel('Accuracy Values (%)')
        plt.xlabel(xlabels[i])
        plt.xticks(x, ('1', '2', '3', '4'))
        fig.savefig('accuracy_summary_{}.png'.format(
            labels[i]), bbox_inches='tight', pad_inches=0)


def t_nse_result_all():
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            d1, d2, d3 = data.get_full_and_partial_dataset(
                config, config.dir_b)
            df = pd.concat([d1, d2, d3], axis=0)

            colors = ['red', 'cyan', 'm', 'yellow',
                      'blue', 'green', 'Purple', 'orange']

            true_x_list, true_y_list = [], []
            # fig, ax = plt.subplots()
            for i, value in enumerate(config.selected_column):
                orig_d = df[df[value] == 1][config.d_16_col]
                x_embedded = TSNE(n_components=2).fit_transform(orig_d)
                gen_x, gen_y = zip(*x_embedded)
                # ax.scatter(gen_x, gen_y, c=colors[i * 2], marker="o", label="ground truth - {}".format(value),
                #            alpha=0.3)
                true_x_list.append(gen_x)
                true_y_list.append(gen_y)
            # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            # fig.savefig(
            #     "{}/{}-tnse-{}-together-{}".format('tsne1-sum', config.super_base.split('/')[-1], direction, 'xxx'),
            #     bbox_inches='tight')

            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            for i in range(len(result_folders)):
                sub_dir = result_folders[i]
                # data_loader, generator, columns
                val_a_fs = os.path.join(
                    config.result_dir, sub_dir, "dataset_a_val.csv")
                a_val_df = pd.read_csv(val_a_fs)[
                    config.d_16_col + config.selected_column]

                train_a_fs = os.path.join(
                    config.result_dir, sub_dir, "dataset_a_train.csv")
                a_train_df = pd.read_csv(train_a_fs)[
                    config.d_16_col + config.selected_column]
                a_train_df = a_train_df[a_train_df[config.validation_feature[0]] == 0]

                data_loader = get_data_loader_for_classifier(
                    [a_val_df, a_train_df], config, False)
                config.model_dir = os.path.join(config.result_dir, sub_dir)
                gan_trainer = GanTrainer2(config,
                                          None,
                                          None,
                                          None)
                generated_df = get_generated_data(data_loader, gan_trainer.G_AB,
                                                  config.d_16_col + config.selected_column)
                gen_x_list, gen_y_list = [], []
                for inx, value in enumerate(config.selected_column):
                    orig_d = generated_df[generated_df[value]
                                          == 1][config.d_16_col]
                    x_embedded = TSNE(n_components=2).fit_transform(orig_d)
                    gen_x, gen_y = zip(*x_embedded)
                    gen_x_list.append(gen_x)
                    gen_y_list.append(gen_y)

                for index, value in enumerate(config.selected_column):
                    fig, ax = plt.subplots()
                    true_x, true_y = true_x_list[index], true_y_list[index]
                    ax.scatter(true_x, true_y, c=colors[index * 2], marker="o", label="ground truth",
                               alpha=0.3)
                    # fig.savefig("{}-tnse-{}-true".format(config.super_base, direction))
                    gen_x, gen_y = gen_x_list[index], gen_y_list[index]
                    ax.scatter(gen_x, gen_y, c=colors[index * 2 + 1], marker='s', label="generation",
                               alpha=0.3)
                    # ax.legend()
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    fig.savefig(
                        "{}/{}-tnse-{}-together-{}-{}".format('tsne1-sum', config.super_base.split('/')[-1], direction,
                                                              i, value), bbox_inches='tight')


def lower_bound():
    prefix = 'backup3'
    direction_list = ['a_to_b', 'b_to_a']
    config_list = [get_summer_spring(prefix, True), get_spring_win(
        prefix, True), get_sum_win(prefix, True)]
    for config in config_list:
        for direction in direction_list:
            #         if config == config_list[0] and direction=='a_to_b':
            #             continue
            do_common(config, direction)
            result_folders = [x for x in os.listdir(
                config.result_dir) if path.isdir(path.join(config.result_dir, x))]
            result_list = []
            for sub_dir in result_folders:
                print('Get df result d from [{}]'.format(sub_dir))
                result_list.append(get_lower_bound(sub_dir, config))
            matrix = get_matrix(result_list)

            result = np.average(matrix, axis=0)
            sum_s = np.sum(result, axis=0)
            res_mt = result / sum_s
            show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'lower_bound'),
                                config.selected_column)
            # result = np.max(matrix, axis=0)
            # sum_s = np.sum(result, axis=0)
            # res_mt = result / sum_s
            # show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'lower_spread'),
            #                     config.selected_column)
            # print(res_mt)
            # result = np.min(matrix, axis=0)
            # sum_s = np.sum(result, axis=0)
            # res_mt = result / sum_s
            # show_and_save_image(res_mt, "{}-{}-{}.png".format(config.super_base, direction, 'higher_spread'),
            #                     config.selected_column)


def reprint_all_matrix():
    sum_spring_a = [[0.385, 0.023, 0.013, 0.255],
                    [0.001, 0.626, 0.107, 0.040],
                    [0.328, 0.021, 0.545, 0.076],
                    [0.286, 0.330, 0.336, 0.629]]
    sum_spring_b = [[0.189, 0.000, 0.012, 0.255],
                    [0.236, 0.710, 0.001, 0.040],
                    [0.108, 0.270, 0.986, 0.076],
                    [0.466, 0.019, 0.001, 0.629]]
    sum_spring_c = [[0.945, 0.000, 0.002, 0.201],
                    [0.000, 0.988, 0.000, 0.000],
                    [0.011, 0.012, 0.996, 0.001],
                    [0.043, 0.00, 0.002, 0.798]]
    sum_spring_d = [[0.498, 0.031, 0.038, 0.170],
                    [0.048, 0.419, 0.123, 0.033],
                    [0.064, 0.103, 0.380, 0.138],
                    [0.390, 0.446, 0.467, 0.658]]
    sum_spring_e = [[0.166, 0.030, 0.000, 0.170],
                    [0.258, 0.693, 0.017, 0.033],
                    [0.151, 0.209, 0.853, 0.138],
                    [0.425, 0.068, 0.130, 0.658]]
    sum_spring_f = [[0.745, 0.000, 0.003, 0.255],
                    [0.080, 0.844, 0.060, 0.000],
                    [0.127, 0.098, 0.906, 0.217],
                    [0.048, 0.058, 0.031, 0.528]]
    sum_spring = [[sum_spring_a, sum_spring_b, sum_spring_c],
                  [sum_spring_d, sum_spring_e, sum_spring_f]]

    sum_win_a = [[0.588, 0.018, 0.013, 0.342],
                 [0.081, 0.852, 0.008, 0.062],
                 [0.164, 0.027, 0.885, 0.104],
                 [0.168, 0.104, 0.094, 0.492]]
    sum_win_b = [[0.176, 0.011, 0.002, 0.342],
                 [0.578, 0.875, 0.011, 0.062],
                 [0.028, 0.037, 0.907, 0.104],
                 [0.217, 0.077, 0.079, 0.492]]
    sum_win_c = [[0.788, 0.003, 0.002, 0.064],
                 [0.003, 0.909, 0.002, 0.018],
                 [0.088, 0.034, 0.967, 0.121],
                 [0.122, 0.053, 0.030, 0.797]]
    sum_win_d = [[0.568, 0.017, 0.060, 0.299],
                 [0.042, 0.929, 0.075, 0.139],
                 [0.379, 0.054, 0.830, 0.095],
                 [0.011, 0.000, 0.034, 0.467]]
    sum_win_e = [[0.599, 0.012, 0.000, 0.299],
                 [0.135, 0.912, 0.001, 0.139],
                 [0.110, 0.027, 0.562, 0.095],
                 [0.156, 0.049, 0.473, 0.467]]
    sum_win_f = [[0.593, 0.027, 0.043, 0.010],
                 [0.032, 0.923, 0.072, 0.007],
                 [0.368, 0.050, 0.853, 0.535],
                 [0.007, 0.000, 0.033, 0.448]]
    sum_win = [[sum_win_a, sum_win_b, sum_win_c],
               [sum_win_d, sum_win_e, sum_win_f]]
    spring_win_a = [[0.349, 0.038, 0.090, 0.190],
                    [0.005, 0.440, 0.021, 0.000],
                    [0.120, 0.039, 0.391, 0.00],
                    [0.527, 0.484, 0.498, 0.810]]
    spring_win_b = [[0.186, 0.021, 0.004, 0.190],
                    [0.025, 0.427, 0.000, 0.000],
                    [0.033, 0.018, 0.412, 0.000],
                    [0.755, 0.524, 0.583, 0.810]]
    spring_win_c = [[0.703, 0.037, 0.039, 0.114],
                    [0.063, 0.931, 0.011, 0.013],
                    [0.064, 0.026, 0.942, 0.027],
                    [0.170, 0.006, 0.008, 0.846]]
    spring_win_d = [[0.342, 0.002, 0.018, 0.189],
                    [0.122, 0.943, 0.022, 0.190],
                    [0.522, 0.049, 0.917, 0.018],
                    [0.015, 0.006, 0.042, 0.603]]
    spring_win_e = [[0.063, 0.007, 0.001, 0.189],
                    [0.384, 0.817, 0.000, 0.190],
                    [0.123, 0.145, 0.999, 0.018],
                    [0.430, 0.032, 0.000, 0.603]]
    spring_win_f = [[0.670, 0.006, 0.054, 0.005],
                    [0.005, 0.972, 0.006, 0.006],
                    [0.300, 0.016, 0.871, 0.054],
                    [0.026, 0.006, 0.068, 0.935]]
    spring_win = [[spring_win_a, spring_win_b, spring_win_c],
                  [spring_win_d, spring_win_e, spring_win_f]]
    suffix = ['only_d', '', 'truth']
    direction = ['a_to_b', 'b_to_a']
    transit = ['sum_spring', 'sum_win', 'spring_win']
    labels = [
        [
            [
                ['Sp1', 'Sp2', 'Sp3', 'SuSp4'],
                ['SuSp1', 'SuSp2', 'SuSp3', 'SuSp4'],
                ['Sp1', 'Sp2', 'Sp3', 'Sp4']
            ],
            [
                ['Su1', 'Su2', 'Su3', 'SpSu4'],
                ['SpSu1', 'SpSu2', 'SpSu3', 'SpSu4'],
                ['Su1', 'Su2', 'Su3', 'Su4']
            ]
        ],
        [
            [
                ['Win1', 'Win2', 'Win3', 'SuWin4'],
                ['SuWin1', 'SuWin2', 'SuWin3', 'SuWin4'],
                ['Win1', 'Win2', 'Win3', 'Win4']
            ],
            [
                ['Su1', 'Su2', 'Su3', 'WinSu4'],
                ['WinSu1', 'WinSu2', 'WinSu3', 'WinSu4'],
                ['Su1', 'Su2', 'Su3', 'Su4']
            ]
        ],
        [
            [
                ['Win1', 'Win2', 'Win3', 'SpWin4'],
                ['SpWin1', 'SpWin2', 'SpWin3', 'SpWin4'],
                ['Win1', 'Win2', 'Win3', 'Win4']
            ],
            [
                ['Sp1', 'Sp2', 'Sp3', 'WinSp4'],
                ['WinSp1', 'WinSp2', 'WinSp3', 'WinSp4'],
                ['Sp1', 'Sp2', 'Sp3', 'Sp4']
            ]
        ]
    ]

    seasoned_data = [sum_spring, sum_win, spring_win]
    for i, seasons in enumerate(seasoned_data):
        name_part_1 = transit[i]
        for j, directed_test in enumerate(seasons):
            name_part_2 = direction[j]
            for k, result in enumerate(directed_test):
                show_bar = k == len(directed_test) - 1
                name_part_3 = suffix[k]
                pattern = 'result/{}-{}-{}' if name_part_3 != '' else 'result/{}-{}'
                res_mat = np.array(result)
                show_and_save_image(res_mat, pattern.format(name_part_1, name_part_2, name_part_3), labels[i][j][k],
                                    show_bar)
