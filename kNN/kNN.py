from numpy import *
import operator
import matplotlib.pyplot as plt


class kNN_class:
    def create_dataset(self) -> [array, list]:
        group = array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
        labels = ['A', 'A', 'B', 'B']
        return group, labels

    def classify0(self, inx: array, data_set: array, labels: list, k: int) -> int:
        data_set_size = data_set.shape[0]
        diff_mat = tile(inx, (data_set_size, 1)) - data_set
        sqdiff_mat = diff_mat**2
        sqdiff_sum = sqdiff_mat.sum(axis=1)
        diff = sqdiff_sum**0.5
        sorted_diff_index = diff.argsort()
        class_cnt = {}
        for i in range(k):
            vote_index = labels[sorted_diff_index[i]]
            class_cnt[vote_index] = class_cnt.get(vote_index, 0) + 1
        sorted_class_cnt = sorted(class_cnt.items(),
                                  key=operator.itemgetter(1), reverse=True)
        return sorted_class_cnt[0][0]

    def file2matrix(self, file_name: str) -> [array, list]:
        with open(file_name) as file:
            array_lines = file.readlines()
            num_of_lines = len(array_lines)
            return_mat = zeros((num_of_lines, 3))
            class_label_vector = []
            index = 0
            for line in array_lines:
                line = line.strip()
                list_from_line = line.split('\t')
                return_mat[index, :] = list_from_line[0:3]
                class_label_vector.append(int(list_from_line[-1]))
                index += 1

            return return_mat, class_label_vector

    def data_show(self, data_set: array, labels:list):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data_set[:, 2], data_set[:, 0],
                   15.0*array(labels), 15.0*array(labels))
        plt.show()

    def auto_norm(self, data_set: array) -> [array, int, int]:
        min_vals = data_set.min(0)
        max_vals = data_set.max(0)
        ranges = max_vals - min_vals
        norm_data_set = zeros(shape(data_set))
        m = data_set.shape[0]
        norm_data_set = data_set - tile(min_vals, (m, 1))
        norm_data_set = norm_data_set/tile(ranges, (m, 1))
        return norm_data_set, ranges, min_vals

    def dating_class_test(self):
        ho_ratio = 0.10
        dating_data_mat, dating_labels = self.file2matrix('datingTestSet.txt')
        norm_mat, ranges, min_vals = self.auto_norm(dating_data_mat)
        m = norm_mat.shape[0]
        num_test_vecs = int(m*ho_ratio)
        error_cnt = 0
        for i in ranges(num_test_vecs):
            class_res = self.classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                  dating_labels[num_test_vecs:m], 3)
            print("the classifier came back with: %d, the real answer is: %d"
                  % (class_res, dating_labels[i]))
            if(class_res != dating_labels[i]):
                error_cnt += 1.0
            print("the total error rate is: %f" % (error_cnt/float(num_test_vecs)))

