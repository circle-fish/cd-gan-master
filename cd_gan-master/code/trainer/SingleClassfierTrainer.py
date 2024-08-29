from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from code.model.classifier import FcClassifier


class SingleClassfierTrainer(object):
    def __init__(self, config, file_path, model_dir, data_loader):
        self.config = config
        self.model_dir = model_dir
        self.file_path = file_path
        self.data_loader = data_loader
        self.classifier = None
        self.build_model()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.try_move_to_gpu()
        if config.load_path:
            self.load_model()

    def train(self):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(chain(self.classifier.parameters()),
                               lr=self.config.learning_rate,
                               betas=(self.config.beta1, self.config.beta2),
                               weight_decay=self.config.weight_decay,
                               amsgrad=self.config.amsgrad)
        # optim.ASGD(chain(self.classifier_a.parameters()))
        #         Adam(chain(self.classifier_a.parameters()),
        #                                lr=self.config.learning_rate,
        #                                betas=(self.config.beta1, self.config.beta2),
        #                                weight_decay=self.config.weight_decay,
        #                                amsgrad=self.config.amsgrad)
        for epoch in range(self.config.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.data_loader, 0):
                # get the inputs
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.classifier(x)
                # print("output", outputs.size())

                #                 labels_cat = torch.argmax(y, dim=1)
                # print("labels_cat", labels_cat.size())
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            if (epoch + 1) % 4 == 0:
                print('classifiecr: [%d] loss: %.3f' %
                      (epoch + 1, running_loss))
            if epoch == self.config.epoch - 1:
                self.print_accuracy()
        self.save_model()

    def build_model(self):
        # self.classifier = FcClassifier([16, 64, 64, 64, len(self.config.selected_column)])
        c_in_length = len(self.config.d_16_col)
        c_out_length = len(self.config.selected_column)
        # total_len = c_in_length + c_out_length
        self.classifier = FcClassifier(
            [c_in_length, c_in_length * 2, c_in_length * 4, c_in_length * 2, c_out_length])

    def load_model(self):
        print("[*] Load models from {}...".format(self.model_dir))
        if self.config.num_gpu == 0:
            def map_location(storage, loc): return storage
        else:
            map_location = None

        filename = '{}/{}.pth'.format(self.model_dir, self.file_path)
        self.classifier.load_state_dict(
            torch.load(filename, map_location=map_location))
        print("[*] Model loaded: {}".format(filename))

    def save_model(self):
        torch.save(self.classifier.state_dict(),
                   '{}/{}.pth'.format(self.model_dir, self.file_path))

    def models(self):
        return self.classifier

    def classify(self, data):
        return self.classifier(data).detach()

    def try_move_to_gpu(self):
        if self.config.num_gpu == 1:
            self.classifier.to(self.device)
        elif self.config.num_gpu > 1:
            self.classifier = nn.DataParallel(
                self.classifier_a.cuda(), device_ids=range(self.num_gpu))

    def _get_variable(self, inputs):
        return inputs.to(self.device)

    def print_accuracy(self):
        len_class = len(self.config.selected_column)
        class_correct = list(0. for i in range(len_class))
        class_total = list(0. for i in range(len_class))
        label_name = self.config.selected_column
        with torch.no_grad():
            for data in self.data_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.classifier(images)
                _, predicted = torch.max(outputs, 1)
                labels_cat = torch.argmax(labels, 1)
                c = (predicted == labels_cat).squeeze()
                for i in range(len(labels_cat)):
                    # print(i)
                    label_cat = labels_cat[i]
                    class_correct[label_cat] += c[i].item()
                    class_total[label_cat] += 1

        for i in range(len(label_name)):
            total = 1 if class_total[i] == 0 else class_total[i]
            print('Accuracy of %5s : %2d %%' % (
                label_name[i], 100 * class_correct[i] / total))


class SingleClassifierTrainer2(object):
    def __init__(self, config, file_path, model_dir, train_data_loader, val_data_loader):
        self.config = config
        self.model_dir = model_dir
        self.file_path = file_path
        self.data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.classifier = None
        self.build_model()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.try_move_to_gpu()
        if config.load_path:
            self.load_model()

    def train(self):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(chain(self.classifier.parameters()),
                               lr=self.config.learning_rate,
                               betas=(self.config.beta1, self.config.beta2),
                               weight_decay=self.config.weight_decay,
                               amsgrad=self.config.amsgrad)
        # optim.ASGD(chain(self.classifier_a.parameters()))
        #         Adam(chain(self.classifier_a.parameters()),
        #                                lr=self.config.learning_rate,
        #                                betas=(self.config.beta1, self.config.beta2),
        #                                weight_decay=self.config.weight_decay,
        #                                amsgrad=self.config.amsgrad)
        for epoch in range(self.config.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.data_loader, 0):
                # get the inputs
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.classifier(x)
                # print("output", outputs.size())

                #                 labels_cat = torch.argmax(y, dim=1)
                # print("labels_cat", labels_cat.size())
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            if (epoch + 1) % 4 == 0:
                print('classifiecr: [%d] loss: %.3f' %
                      (epoch + 1, running_loss))
            if epoch == self.config.epoch - 1:
                self.print_accuracy()
        self.save_model()

    def build_model(self):
        # self.classifier = FcClassifier([16, 64, 64, 64, len(self.config.selected_column)])
        c_in_length = len(self.config.d_16_col)
        c_out_length = len(self.config.selected_column)
        # total_len = c_in_length + c_out_length
        self.classifier = FcClassifier(
            [c_in_length, c_in_length * 2, c_in_length * 4, c_in_length * 2, c_out_length])

    def load_model(self):
        print("[*] Load models from {}...".format(self.model_dir))
        if self.config.num_gpu == 0:
            def map_location(storage, loc): return storage
        else:
            map_location = None

        filename = '{}/{}.pth'.format(self.model_dir, self.file_path)
        self.classifier.load_state_dict(
            torch.load(filename, map_location=map_location))
        print("[*] Model loaded: {}".format(filename))

    def save_model(self):
        torch.save(self.classifier.state_dict(),
                   '{}/{}.pth'.format(self.model_dir, self.file_path))

    def models(self):
        return self.classifier

    def classify(self, data):
        return self.classifier(data).detach()

    def try_move_to_gpu(self):
        if self.config.num_gpu == 1:
            self.classifier.to(self.device)
        elif self.config.num_gpu > 1:
            self.classifier = nn.DataParallel(
                self.classifier_a.cuda(), device_ids=range(self.num_gpu))

    def _get_variable(self, inputs):
        return inputs.to(self.device)

    def print_accuracy(self):
        len_class = len(self.config.selected_column)
        class_correct = list(0. for i in range(len_class))
        class_total = list(0. for i in range(len_class))
        label_name = self.config.selected_column
        with torch.no_grad():
            for data in self.val_data_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.classifier(images)
                _, predicted = torch.max(outputs, 1)
                labels_cat = torch.argmax(labels, 1)
                c = (predicted == labels_cat).squeeze()
                for i in range(len(labels_cat)):
                    # print(i)
                    label_cat = labels_cat[i]
                    class_correct[label_cat] += c[i].item()
                    class_total[label_cat] += 1

        for i in range(len(label_name)):
            total = 1 if class_total[i] == 0 else class_total[i]
            print('Accuracy of %5s : %2d %%' % (
                label_name[i], 100 * class_correct[i] / total))
