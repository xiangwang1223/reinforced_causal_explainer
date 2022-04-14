# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:44:39 2020

@author: Lenovo
"""
import os
import os.path as osp

import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data


class Reddit5k(InMemoryDataset):

    url = ('https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/REDDIT-MULTI-5K.zip')

    splits = ['training', 'evaluation', 'testing']
    def __init__(self, root, mode='testing', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        super(Reddit5k, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):

        return ['REDDIT-MULTI-5K/' + i \
                for i in [
                    'REDDIT-MULTI-5K_A.txt',
                    'REDDIT-MULTI-5K_graph_indicator.txt',
                    'REDDIT-MULTI-5K_graph_labels.txt'
                    ]
                ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'evaluation.pt', 'testing.pt']

    def download(self):

        if os.path.exists(osp.join(self.raw_dir, 'REDDIT-MULTI-5K')):
            print('Using existing data in folder REDDIT-MULTI-5K')
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        edge_index = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=',').T
        edge_index = torch.from_numpy(edge_index - 1.0).to(torch.long)  # node idx from 0

        z = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]), dtype=int)

        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[-1])) - 1.
        y = torch.unsqueeze(torch.tensor(y), 1).long()


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

            g_edge_index = edge_index[:, begin:end] - int(min(perm))
            row, col = g_edge_index

            x = torch.zeros((len(perm), 1)).float()
            # degree feature as node feature
            for i in range(x.size(0)):
                x[i, 0] = (row == i).sum()

            # generate constant labels for edges
            edge_attr = torch.ones((g_edge_index.size(1), 1)).float()

            data = Data(x=x, y=y[i],
                        edge_index=g_edge_index,
                        edge_attr=edge_attr,
                        name="reddit_%d" % i, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            begin = end
            data_list.append(data)

        assert len(data_list) == 4999

        random.shuffle(data_list)
        torch.save(self.collate(data_list[1000:]), self.processed_paths[0])
        torch.save(self.collate(data_list[500:1000]), self.processed_paths[1])
        torch.save(self.collate(data_list[:500]), self.processed_paths[2])