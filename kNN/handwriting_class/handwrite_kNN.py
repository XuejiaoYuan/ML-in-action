from numpy import *
from os import listdir
from kNN.kNN import kNN_class
BASE_DIR = 'kNN/handwriting_class/digits/'


def img2vector(filename: str) -> list:
    return_vec = zeros((1, 1024))
    with open(filename) as file:
        for i in range(32):
            lins_str = file.readline()
            for j in range(32):
                return_vec[0, 32*i+j] = int(lins_str[j])
    return return_vec


def handwriting_class_test():
    knn = kNN_class()
    hwLabels = []
    training_file_list = listdir('digits/trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        filename_str = training_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hwLabels.append(class_num_str)
        training_mat[i, :] = img2vector('digits/trainingDigits/%s' % filename_str)
    test_file_list = listdir('digits/testDigits')
    error_cnt = 0.0
    mTest = len(test_file_list)
    for i in range(mTest):
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('digits/testDigits/%s' % filename_str)
        classifier_res = knn.classify0(vector_under_test, training_mat, hwLabels, 3)
        print('the classifier came back with: %d, the real, the real answer is: %d' % (classifier_res, class_num_str))
        if classifier_res != class_num_str:
            error_cnt += 1.0
        print('\nthe total number of errors is : %d' % error_cnt)
        print('\nthe total error rate is : %f' % (error_cnt / float(mTest)))

handwriting_class_test()