'''
用来读取 cifar10 数据集
'''
import pickle
import re
import os
import numpy as np
import matplotlib.pyplot as plt

class Cifar10:
    def __init__(self,path,one_hot = True):
        self.path = path
        self.one_hot = one_hot
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 50000

    def _load_data(self):
        images = []
        labels = []
        files = os.listdir(self.path)
        for file in files:
            if re.match('data_batch_*',file):
                with open(os.path.join(self.path,file),'rb') as fo:
                    data = pickle.load(fo,encoding='bytes')
                    images.append(data[b'data'].reshape([-1,3,32,32]))
                    labels.append(data[b'labels'])
            elif re.match('test_batch',file):
                with open(os.path.join(self.path,file),'rb') as fo:
                    data = pickle.load(fo,encoding='bytes')
                    test_images = np.array(data[b'data'].reshape([-1,3,32,32]))
                    test_labels = np.array(data[b'labels'])
        images = np.concatenate(images,axis = 0)
        labels = np.concatenate(labels,axis = 0)
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self.train_images = images.transpose(0,2,3,1)[perm]
        self.train_labels = np.array(labels).reshape([-1,1])[perm]
        self.test_images = test_images.transpose(0,2,3,1)
        self.test_labels = test_labels.reshape([-1, 1])
        if self.one_hot:
            self.train_labels = self._one_hot(self.train_labels,10)
            self.test_labels = self._one_hot(self.test_labels,10)

        return self.train_images,self.train_images,self.test_images,self.test_labels

    def next_batch(self,batch_size,shuffle=True):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.train_images = self.train_images[perm]
                self.train_labels = self.train_labels[perm]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.train_images[start:end],self.train_labels[start:end]

    def _one_hot(self,labels,num):
        size = labels.shape[0]
        label_one_hot = np.zeros([size,num])
        for i in range(size):
            label_one_hot[i,np.squeeze(labels[i])] = 1
        return label_one_hot

def load_cifar10(path,one_hot = False):
    cifar10 = Cifar10(path,one_hot)
    cifar10._load_data()
    return cifar10

if __name__ == '__main__':
    path = 'd:/input_data/cifar-10/cifar-10-batches-py/'
    cifar10 = load_cifar10(path,one_hot = False)
    images = cifar10.train_images
    labels = cifar10.train_labels
    test_images = cifar10.test_images
    test_labels = cifar10.test_labels
    print("训练集shape = ",images.shape )
    print('测试集shape = ',test_images.shape)
    batch_xs,batch_ys = cifar10.next_batch(batch_size = 64,shuffle=True)
    print("batch_xs shape = ",batch_xs.shape)
    print("batch_ys shape = ",batch_ys.shape)


    # plot image
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    num_classes = len(classes)
    samples_per_class = 7
    for y, clss in enumerate(classes):
        idxs = np.flatnonzero(labels == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(images[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(clss)
    plt.show()

    # batch中的信息：{'num_cases_per_batch': 10000, 'label_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'num_vis': 3072}
    # cifar10_name = np.load(os.path.join(path,'batches.meta'))
    # print(cifar10_name)







