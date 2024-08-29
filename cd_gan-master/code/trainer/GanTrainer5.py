import os
import shutil
import time
from glob import glob
from itertools import chain
from os import path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from code.config import Config
from code.model.classifier import FcClassifier
from code.model.discriminators import FcDiscriminator
from code.model.generators import FcGeneratorReLu
from code.utils import data


class DataLoaders(object):
    def __init__(self, b_train_loader, a_train_loader, d_train_loader, b_val_loader, a_val_loader):
        self.b_train_loader: DataLoader = b_train_loader
        self.a_train_loader: DataLoader = a_train_loader
        self.d_train_loader: DataLoader = d_train_loader
        self.b_val_loader: DataLoader = b_val_loader
        self.a_val_loader: DataLoader = a_val_loader

    def save(self, base_dir):
        data.save(self.b_train_loader.dataset, path.join(base_dir, 'b_train_loader.csv'))
        data.save(self.a_train_loader.dataset, path.join(base_dir, 'a_train_loader.csv'))
        data.save(self.d_train_loader.dataset, path.join(base_dir, 'd_train_loader.csv'))
        data.save(self.b_val_loader.dataset, path.join(base_dir, 'b_val_loader.csv'))
        data.save(self.a_val_loader.dataset, path.join(base_dir, 'a_val_loader.csv'))

    def load(self, config: Config, base_dir: str, shuffle: bool):
        self.b_train_loader = data.get_data_loader_from_file(
            path.join(base_dir, 'b_train_loader.csv'), config, shuffle)
        self.a_train_loader = data.get_data_loader_from_file(
            path.join(base_dir, 'a_train_loader.csv'), config, shuffle)
        self.d_train_loader = data.get_data_loader_from_file(
            path.join(base_dir, 'd_train_loader.csv'), config, shuffle)
        self.b_val_loader = data.get_data_loader_from_file(
            path.join(base_dir, 'b_val_loader.csv'), config, shuffle)
        self.a_val_loader = data.get_data_loader_from_file(
            path.join(base_dir, 'a_val_loader.csv'), config, shuffle)


class GanTrainer5(object):
    def __init__(self, config: Config, data_loaders: DataLoaders, do_disco_gan_test_internal, do_ground_truth_test,
                 b_ground_truth_classifier):
        self.do_disco_gan_test_internal = do_disco_gan_test_internal
        self.do_ground_truth_test = do_ground_truth_test
        self.b_ground_truth_classifier = b_ground_truth_classifier
        self.config = config
        self.data_loaders = data_loaders
        self.G_AB, self.G_BA, self.C_A, self.C_B, self.D_A, self.D_B = None, None, None, None, None, None
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.G_AB.to(self.device)
        self.G_BA.to(self.device)
        self.C_A.to(self.device)
        self.C_B.to(self.device)
        self.D_A.to(self.device)
        self.D_B.to(self.device)
        if config.load_path:
            self.load_model()

    def build_model(self):
        c_in_length = len(self.config.d_16_col)
        c_out_length = len(self.config.selected_column)
        total_len = c_in_length + c_out_length
        self.G_AB = FcGeneratorReLu(total_len, total_len,
                                    [total_len * 2, total_len * 4, total_len * 8, total_len * 4, total_len * 2])
        self.G_BA = FcGeneratorReLu(total_len, total_len,
                                    [total_len * 2, total_len * 4, total_len * 8, total_len * 4, total_len * 2])
        self.C_A = FcClassifier([c_in_length, c_in_length * 2, c_in_length * 4, c_in_length * 2, c_out_length])
        self.C_B = FcClassifier([c_in_length, c_in_length * 2, c_in_length * 4, c_in_length * 2, c_out_length])
        self.D_A = FcDiscriminator(total_len, 1, [total_len * 2, total_len * 4, total_len * 2])
        self.D_B = FcDiscriminator(total_len, 1, [total_len * 2, total_len * 4, total_len * 2])

    def load_model(self):
        paths = glob(os.path.join(self.config.model_dir, 'G_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.config.model_dir))
            return
        idx_st = [os.path.basename(path).split('.')[0].split('_')[-1] for path in paths]
        #         print(idx_st)
        idxes = [int(st) for st in idx_st if st.isdigit()]
        self.config.start_step = max(idxes)

        if self.config.num_gpu == 0:
            def map_location(storage, loc): return storage
        else:
            map_location = None

        g_ab_filename = '{}/G_AB_{}.pth'.format(self.config.model_dir, self.config.start_step)
        self.G_AB.load_state_dict(torch.load(g_ab_filename, map_location=map_location))
        print("[*] Model loaded: {}".format(g_ab_filename))

        g_ba_filename = '{}/G_BA_{}.pth'.format(self.config.model_dir, self.config.start_step)
        self.G_BA.load_state_dict(torch.load(g_ba_filename, map_location=map_location))
        print("[*] Model loaded: {}".format(g_ba_filename))

        c_a_filename = '{}/C_A_{}.pth'.format(self.config.model_dir, self.config.start_step)
        self.C_A.load_state_dict(torch.load(c_a_filename, map_location=map_location))
        print("[*] Model loaded: {}".format(c_a_filename))

        c_b_filename = '{}/C_B_{}.pth'.format(self.config.model_dir, self.config.start_step)
        self.C_B.load_state_dict(torch.load(
            c_b_filename, map_location=map_location))
        print("[*] Model loaded: {}".format(c_b_filename))

        d_a_filename = '{}/D_A_{}.pth'.format(self.config.model_dir, self.config.start_step)
        self.D_A.load_state_dict(torch.load(
            d_a_filename, map_location=map_location))
        print("[*] Model loaded: {}".format(d_a_filename))

        d_b_filename = '{}/D_B_{}.pth'.format(self.config.model_dir, self.config.start_step)
        self.D_B.load_state_dict(torch.load(
            d_b_filename, map_location=map_location))
        print("[*] Model loaded: {}".format(d_b_filename))

    def save_model(self, step):
        g_ab_filename = '{}/G_AB_{}.pth'.format(self.config.model_dir, step)
        torch.save(self.G_AB.state_dict(), g_ab_filename)
        print("[*] Model saved: {}".format(g_ab_filename))

        g_ba_filename = '{}/G_BA_{}.pth'.format(self.config.model_dir, step)
        torch.save(self.G_BA.state_dict(), g_ba_filename)
        print("[*] Model saved: {}".format(g_ba_filename))

        c_a_filename = '{}/C_A_{}.pth'.format(self.config.model_dir, step)
        torch.save(self.C_A.state_dict(), c_a_filename)
        print("[*] Model saved: {}".format(c_a_filename))

        c_b_filename = '{}/C_B_{}.pth'.format(self.config.model_dir, step)
        torch.save(self.C_B.state_dict(), c_b_filename)
        print("[*] Model saved: {}".format(c_b_filename))

        d_a_filename = '{}/D_A_{}.pth'.format(self.config.model_dir, step)
        torch.save(self.D_A.state_dict(), d_a_filename)
        print("[*] Model saved: {}".format(d_a_filename))

        d_b_filename = '{}/D_B_{}.pth'.format(self.config.model_dir, step)
        torch.save(self.D_B.state_dict(), d_b_filename)
        print("[*] Model saved: {}".format(d_b_filename))

    def train(self):
        c_in_length = len(self.config.d_16_col)
        c_out_length = len(self.config.selected_column)
        one = torch.FloatTensor(self.config.batch_size)
        _ = one.data.fill_(1)

        zero = torch.FloatTensor(self.config.batch_size)
        _ = zero.data.fill_(0)

        mse = nn.MSELoss()
        bce = mse
        # nn.BCELoss()
        cel = nn.CrossEntropyLoss()
        dd = nn.MSELoss(reduction='sum')

        # if self.config.num_gpu > 0:
        #     mse.cuda()
        #     bce.cuda()
        #     cel.cuda()
        #     one = one.cuda()
        #     zero = zero.cuda()
        optimizer = torch.optim.Adam

        optimizer_g = optimizer(
            chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        optimizer_c = optimizer(
            chain(self.C_A.parameters(), self.C_B.parameters()),
            lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        optimizer_d = optimizer(
            chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))

        step = self.config.start_step
        while True:
            b_loader = iter(self.data_loaders.b_train_loader)
            a_loader = iter(self.data_loaders.a_train_loader)
            d_loader = iter(self.data_loaders.d_train_loader)

            # if step % self.config.ratio_in_total < self.config.ratio_d_thresh:
            if step % 2 == 0:
                # print('###########################')
                # print('step[{}/{}]:'.format(step, self.config.max_step))
                # print('###########################')
                # while _ in self.data_loaders:
                # Do normal training
                x_b, y_b = b_loader.next()
                x_a, y_a = a_loader.next()
                x_a, y_a, x_b, y_b = self._get_variable(x_a), self._get_variable(y_a), \
                                     self._get_variable(x_b), self._get_variable(y_b)
                in_a = torch.cat((x_a, y_a), 1)
                in_b = torch.cat((x_b, y_b), 1)
                self.G_AB.zero_grad()
                self.G_BA.zero_grad()
                x_ab = self.G_AB(in_a)
                self.d_gen = None
                x_ba = self.G_BA(in_b)

                x_aba = self.G_BA(x_ab)
                x_bab = self.G_AB(x_ba)

                l_const_a = mse(x_aba, in_a)
                l_const_b = mse(x_bab, in_b)

                l_cla_ba = bce(self.C_A(x_ba[:, :c_in_length]), y_a)
                l_cla_ab = bce(self.C_B(x_ab[:, :c_in_length]), y_b)
                l_cla_aba = bce(self.C_A(x_aba[:, :c_in_length]), y_a)
                l_cla_bab = bce(self.C_B(x_bab[:, :c_in_length]), y_b)
                # l_const_a + l_const_b +

                l_d_aba = bce(self.D_A(x_aba), one)
                l_d_bab = bce(self.D_A(x_bab), one)
                l_d_ab = bce(self.D_B(x_ab), one)
                l_d_ba = bce(self.D_A(x_ba), one)

                l_c_g = l_cla_ba + l_cla_ab + l_cla_aba + l_cla_bab
                l_d_g = l_d_aba + l_d_bab + l_d_ab + l_d_ba
                l_g = l_c_g + l_d_g

                l_g.backward()
                optimizer_g.step()

                self.C_A.zero_grad()
                self.C_B.zero_grad()
                fake_a = 1 - y_a
                fake_b = 1 - y_b
                x_ab = self.G_AB(in_a)
                x_ba = self.G_BA(in_b)

                x_aba = self.G_BA(x_ab)
                x_bab = self.G_AB(x_ba)
                l_cla_a = bce(self.C_A(in_a[:, :c_in_length]), y_a)
                l_cla_b = bce(self.C_B(in_b[:, :c_in_length]), y_b)
                # if step % self.config.ratio_in_total < self.config.ratio_c_thresh:
                #     l_cla_ba = bce(self.C_A(x_ba[:, :c_in_length]), fake_a)
                #     l_cla_ab = bce(self.C_B(x_ab[:, :c_in_length]), fake_b)
                #     l_cla_aba = bce(self.C_A(x_aba[:, :c_in_length]), fake_a)
                #     l_cla_bab = bce(self.C_B(x_bab[:, :c_in_length]), fake_b)
                #     l_c = l_cla_ba + l_cla_ab + l_cla_aba + l_cla_bab + l_cla_a + l_cla_b
                # else:
                l_c = l_cla_a + l_cla_b

                l_c.backward()
                optimizer_c.step()

                if step % self.config.ratio_in_total < self.config.ratio_c_d_thresh:
                    self.D_A.zero_grad()
                    self.D_B.zero_grad()
                    x_ab = self.G_AB(in_a)
                    x_ba = self.G_BA(in_b)
                    x_aba = self.G_BA(x_ab)
                    x_bab = self.G_AB(x_ba)

                    l_d_a = bce(self.D_A(in_a), one)
                    l_d_b = bce(self.D_B(in_b), one)
                    # l_d_aba = bce(self.D_A(x_aba), zero)
                    # l_d_bab = bce(self.D_A(x_bab), zero)
                    l_d_ab = bce(self.D_B(x_ab), zero)
                    l_d_ba = bce(self.D_A(x_ba), zero)
                    # + l_d_aba + l_d_bab
                    l_d = l_d_a + l_d_b + l_d_ab + l_d_ba
                    l_d.backward()
                    optimizer_d.step()
                else:
                    self.D_A.zero_grad()
                    self.D_B.zero_grad()
                    l_d_a = bce(self.D_A(in_a), one)
                    l_d_b = bce(self.D_B(in_b), one)
                    l_d = l_d_a + l_d_b
                    l_d.backward()
                    optimizer_d.step()
            else:
                # do cat mse training
                x_d, y_d = d_loader.next()
                x_d, y_d = self._get_variable(x_d), self._get_variable(y_d)
                in_a = torch.cat((x_d, y_d), 1)

                self.G_AB.zero_grad()
                self.G_BA.zero_grad()

                x_ab = self.G_AB(in_a)
                x_aba = self.G_BA(x_ab)
                l_const_a = mse(x_aba, in_a)
                l_cla_aba = bce(self.C_A(x_aba[:, :c_in_length]), y_d)
                # print('###########################')
                # print('step:', step)
                # print('classifier_pred_label_d:', self.C_A(x_aba[:, :c_in_length]))
                # print('ground_truth_label_d:', y_d)
                # print('###########################')
                l_cla_ab = bce(self.C_B(x_ab[:, :c_in_length]), y_d)
                # print('###########################')
                # print('step:', step)
                # print('classifier_pred_label_d:', self.C_A(x_ab[:, :c_in_length]))
                # print('ground_truth_label_d:', y_d)
                # print('###########################')
                # l_const_a +
                l_d_aba = bce(self.D_A(x_aba), one)
                l_g = l_cla_ab
                # + l_cla_aba + l_d_aba
                l_g.backward()
                optimizer_g.step()

                # Classifier
                self.C_A.zero_grad()
                self.C_B.zero_grad()

                fake_d = 1 - y_d
                # x_ab = self.G_AB(in_a)
                # x_aba = self.G_BA(x_ab)

                l_cla_a = bce(self.C_A(in_a[:, :c_in_length]), y_d)
                # l_cla_ab = cel(self.C_B(x_ab[:, :c_in_length]), y_d)
                # if step % self.config.ratio_in_total < self.config.ratio_d_d_thresh:
                #     l_cla_a = bce(self.C_A(in_a[:, :c_in_length]), y_d)
                #     l_cla_ab = cel(self.C_B(x_ab[:, :c_in_length]), y_d)
                #     l_cla_aba = bce(self.C_A(x_aba[:, :c_in_length]), fake_d)
                #     l_c = l_cla_a + l_cla_ab
                #     # + l_cla_aba
                # else:
                #     l_cla_a = bce(self.C_A(in_a[:, :c_in_length]), y_d)
                # l_cla_ab = bce(self.C_B(x_ab[:, :c_in_length]), y_d)
                l_c = l_cla_a
                # + l_cla_ab
                l_c.backward()
                optimizer_c.step()

                if step % self.config.ratio_in_total < self.config.ratio_d_d_thresh:
                    self.D_A.zero_grad()
                    self.D_B.zero_grad()
                    x_ab = self.G_AB(in_a)
                    x_aba = self.G_BA(x_ab)

                    l_d_a = bce(self.D_A(in_a), one)
                    l_d_aba = bce(self.D_A(x_aba), zero)
                    l_d_ab = bce(self.D_B(x_ab), zero)
                    l_d = l_d_a + l_d_aba + l_d_ab
                    l_d.backward()
                    optimizer_d.step()

            if step % self.config.log_step == 0:
                print("[{}/{}] Loss_G: {:.4f}".format(step, self.config.max_step, l_g.data))
                print("[{}/{}] Loss_C: {:.4f}".format(step, self.config.max_step, l_c.data))
                print("[{}/{}] Loss_D: {:.4f}".format(step, self.config.max_step, l_d.data))
                # l_const_b is not None and
                # l_const_a is not None and
                if l_cla_ba is not None and l_cla_ab is not None and l_cla_aba is not None and l_cla_bab is not None:
                    print(
                        ("[{}/{}] " +
                         "l_cla_a: {:f}, l_cla_b: {:f} l_cla_aba: {:f}, l_cla_bab: {:f}").format(step,
                                                                                                 self.config.max_step,
                                                                                                 l_cla_ba.data,
                                                                                                 l_cla_ab.data,
                                                                                                 l_cla_aba.data,
                                                                                                 l_cla_bab.data))

            if step % self.config.save_step == self.config.save_step - 1:
                self.save_model(step)
                base_dir = '{}/{:.2f}_{:.0f}'.format(self.config.result_dir, step, time.time())
                os.mkdir(base_dir)
                self.save_everything(base_dir, step)

                # generated d several times for saving
                ti = 0
                DG = []
                while ti < self.config.gen_time:
                    # print('###########################')
                    # print('step:{} | gen_time:{}:'.format(step, ti))
                    # print('###########################')
                    x_ab = self.G_AB(in_a)
                    DG.append(x_ab.detach().numpy())
                    ti += 1
                self.d_gen = np.concatenate(DG, axis=0)
                print('###########################')
                print('step:', step)
                print('d_gen:', self.d_gen.shape)
                print('###########################')

                np.save(path.join(base_dir, 'd_eval_gen.npy'), self.d_gen)
                print('saved {} .npy file in {}'.format(self.d_gen.shape, base_dir))

            step += 1
            if step >= self.config.max_step:
                break

    def _get_variable(self, inputs):
        if self.config.num_gpu > 0 and inputs is not None:
            out = inputs.to(self.device)
        else:
            out = inputs
        return out

    def save_everything(self, base_dir, step):
        g_ab_filename = '{}/G_AB_{}.pth'.format(self.config.model_dir, step)
        g_ba_filename = '{}/G_BA_{}.pth'.format(self.config.model_dir, step)
        c_a_filename = '{}/C_A_{}.pth'.format(self.config.model_dir, step)
        c_b_filename = '{}/C_B_{}.pth'.format(self.config.model_dir, step)
        d_a_filename = '{}/D_A_{}.pth'.format(self.config.model_dir, step)
        d_b_filename = '{}/D_B_{}.pth'.format(self.config.model_dir, step)

        shutil.copyfile(g_ab_filename, path.join(base_dir, 'G_AB_{}.pth'.format(step)))
        shutil.copyfile(g_ba_filename, path.join(base_dir, 'G_BA_{}.pth'.format(step)))
        shutil.copyfile(c_a_filename, path.join(base_dir, 'C_A_{}.pth'.format(step)))
        shutil.copyfile(c_b_filename, path.join(base_dir, 'C_B_{}.pth'.format(step)))
        shutil.copyfile(d_a_filename, path.join(base_dir, 'D_B_{}.pth'.format(step)))
        shutil.copyfile(d_b_filename, path.join( base_dir, 'D_A_{}.pth'.format(step)))

        self.data_loaders.save(base_dir)
