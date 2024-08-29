import torch
import numpy as np


def do_classifier_test(data_loader, classifier, label_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = len(label_name)
    matrix_a = np.zeros((n, n))
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x_a, y_a = data
            x_a, y_a = x_a.to(device), y_a.to(device)
            predict = classifier(x_a).detach()
            predicted = torch.argmax(predict, dim=1).cpu()
            labels_cat = torch.argmax(y_a, dim=1).cpu()
            for inx in list(zip(predicted, labels_cat)):
                matrix_a[inx[0], inx[1]] += 1
    print('accuracy of A')
    print(matrix_a)
    return matrix_a


def do_disco_gan_test_internal(data_loader, generator, classifier_a, label_name, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    class_correct = list(0. for i in range(len(label_name)))
    class_total = list(0. for i in range(len(label_name)))
    idx = 0
    predicted_cnt = list(0 for i in range(len(label_name)))
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x_a, y_a, _, _ = data
            x_a, y_a = x_a.to(device), y_a.to(device)
            input = torch.cat((x_a, y_a), 1)
            outputs = generator(input).detach()
            predict = classifier_a(outputs[:, :len(config.d_16_col)]).detach()
            predicted = torch.argmax(predict, dim=1)
            labels_cat = torch.argmax(y_a, dim=1)
            total += y_a.size(0)
            correct += (predicted == labels_cat).sum().item()
            c = (predicted == labels_cat)
            for i in range(len(labels_cat)):
                label_cat = labels_cat[i]
                predicted_cnt[predicted[i].item()] += 1
                class_correct[label_cat] += c[i].item()
                class_total[label_cat] += 1
            idx = index
    print('Accuracy of test result: %.2f %%' % (100.0 * correct / total))
    for i in range(len(label_name)):
        print('Accuracy of %5s : %.2f %%, %d / %d, pre: %d' % (
            label_name[i], 100 * class_correct[i] / (class_total[i] + 1e-8), class_correct[i], class_total[i],
            predicted_cnt[i]))
    ac1 = 100.0 * correct / total
    return ac1


def do_generate_result_test(data_loader, generator, classifier, label_name, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = len(label_name)
    result_matrix = np.zeros((n, n))
    largest_index = 0
    n_col = len(config.d_16_col)
    with torch.no_grad():
        for index, data in enumerate(data_loader):
            x_a, y_a = data
            x_a, y_a = x_a.to(device), y_a.to(device)
            input = torch.cat((x_a, y_a), 1)
            outputs = generator(input).detach()
            predict = classifier(outputs[:, :n_col]).detach()
            predicted = torch.argmax(predict, dim=1).cpu()
            labels_cat = torch.argmax(y_a, dim=1).cpu()
            for inx in list(zip(predicted, labels_cat)):
                result_matrix[inx[0], inx[1]] += 1
            largest_index = index
    return result_matrix
