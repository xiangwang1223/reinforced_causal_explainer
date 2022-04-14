# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:44:39 2020

@author: Lenovo
"""
import os
import os.path as osp

import torch
import numpy as np
import sklearn.preprocessing as preprocessing
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data


class Tox21(InMemoryDataset):

    url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/Tox21_AhR_{}.zip'
    splits = ['training', 'evaluation', 'testing']  # train/val/test splits.
    num_atomic_species = 53

    def __init__(self, root, mode='testing', transform=None, pre_transform=None,
                 pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        super(Tox21, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):

        return ['Tox21_AhR_{}/'.format(self.mode) + i\
                for i in [
                    'Tox21_AhR_{}_A.txt'.format(self.mode),
                    'Tox21_AhR_{}_edge_labels.txt'.format(self.mode),
                    'Tox21_AhR_{}_graph_indicator.txt'.format(self.mode),
                    'Tox21_AhR_{}_graph_labels.txt'.format(self.mode),
                    'Tox21_AhR_{}_node_labels.txt'.format(self.mode)
                ]
                ]

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.mode)

    def download(self):
        for mode in self.splits:
            folder = 'Tox21_AhR_{}'.format(mode)
            if os.path.exists(osp.join(self.raw_dir, folder)):
                print('Using existing data in folder %s' % folder)
                return

            path = download_url(self.url.format(mode), self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        
        edge_index = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=',').T
        edge_index = torch.from_numpy(edge_index - 1.0).to(torch.long)  # node idx from 0

        edge_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]))
        encoder = preprocessing.OneHotEncoder().fit(np.unique(edge_label).reshape(-1, 1))
        edge_attr = encoder.transform(edge_label.reshape(-1, 1)).toarray()
        edge_attr = torch.Tensor(edge_attr)

        node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[-1]))
        x = torch.zeros((len(node_label), self.num_atomic_species))
        for idx, label in enumerate(node_label):
            x[idx, int(label)] = 1.

        z = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]), dtype=int)

        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]))
        y = torch.unsqueeze(torch.LongTensor(y), 1)
        num_graphs = len(y)
        total_edges = edge_index.size(1)
        begin = 0

        data_list = []
        for i in range(num_graphs):

            perm = np.where(z == i + 1)[0]
            bound = max(perm)
            end = begin
            for end in range(begin, total_edges):
                if int(edge_index[0, end]) > bound:
                    break

            data = Data(x=x[perm], y=y[i], z=node_label[perm],
                        edge_index=edge_index[:, begin:end] - int(min(perm)),
                        edge_attr=edge_attr[begin:end],
                        name="tox21_%d" % i, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            begin = end
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])