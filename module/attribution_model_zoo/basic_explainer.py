import os
import copy
import math
import time
import random
import warnings
import numpy as np
import networkx as nx
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import re
import requests
import scipy as sp
import torch.nn as nn
import torch_geometric
from torch.autograd import Variable

import torch
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image
# from module.visual_genome import api
# from module.visual_genome import local as vgl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ifd_pert = 0.1
n_class_dict = {'MutagNet': 2, 'Tox21Net': 2, 'Reddit5kNet': 5, 'VGNet': 5}
vis_dict = {
    'MutagNet': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'Tox21Net': {'node_size': 400, 'linewidths': 1, 'font_size': 10, 'width': 3},
    'defult': {'node_size': 200, 'linewidths': .1, 'font_size' : 1, 'width': 0.2}
}
chem_graph_label_dict = {'MutagNet': {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                                      8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'},
                         'Tox21Net': {0: 'O', 1: 'C', 2: 'N', 3: 'F', 4: 'Cl', 5: 'S', 6: 'Br', 7: 'Si',
                                      8: 'Na', 9: 'I',10: 'Hg', 11: 'B', 12: 'K', 13: 'P', 14: 'Au',
                                      15: 'Cr', 16: 'Sn', 17: 'Ca', 18: 'Cd',19: 'Zn', 20: 'V', 21: 'As',
                                      22: 'Li', 23: 'Cu', 24: 'Co', 25: 'Ag', 26: 'Se', 27: 'Pt', 28: 'Al',
                                      29: 'Bi', 30: 'Sb', 31: 'Ba', 32: 'Fe', 33: 'H', 34: 'Ti', 35: 'Tl',
                                      36: 'Sr',37: 'In', 38: 'Dy', 39: 'Ni', 40: 'Be', 41: 'Mg', 42: 'Nd',
                                      43: 'Pd', 44: 'Mn', 45: 'Zr', 46: 'Pb', 47: 'Yb', 48: 'Mo', 49: 'Ge',
                                      50: 'Ru', 51: 'Eu', 52: 'Sc'}
                         }
rec_color = ['cyan', 'mediumblue',  'deeppink', 'darkorange',  'gold', 'chartreuse',   'lightcoral','darkviolet', 'teal', 'lightgrey',]

class Explainer(object):

    def __init__(self, gnn_model_path):
        self.model = torch.load(gnn_model_path).to(device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

        self.path = gnn_model_path
        self.last_result = None
        self.vis_dict = None

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    def get_cxplain_scores(self, graph):
        # initialize the ranking list with cxplain.
        y = graph.y
        orig_pred = self.model(graph.x,
                               graph.edge_index,
                               graph.edge_attr,
                               graph.batch)[0, y]

        scores = []
        for e_id in range(graph.num_edges):
            edge_mask = torch.ones(graph.num_edges, dtype=torch.bool)
            edge_mask[e_id] = False
            masked_edge_index = graph.edge_index[:, edge_mask]
            masked_edge_attr = graph.edge_attr[edge_mask]

            masked_pred = self.model(graph.x,
                                     masked_edge_index,
                                     masked_edge_attr,
                                     graph.batch)[0, y]

            scores.append(orig_pred - masked_pred)
            # scores.append(orig_pred - masked_pred)
        scores = torch.tensor(scores)
        return scores.cpu().detach().numpy()

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()
    #@staticmethod
    #def relabel(x, edge_index):

    #    num_nodes = x.size(0)
    #    sub_nodes = torch.unique(edge_index)
        # .... the node features.
    #    x = x[sub_nodes]

    #    row, _ = edge_index

        # remapping the nodes in the explanatory subgraph to new ids.
    #    node_idx = row.new_full((num_nodes,), -1)
    #    row[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    #    edge_index = node_idx[edge_index]
    #    return x, edge_index

    def pack_explanatory_subgraph(self, top_ratio=0.2):

        graph, imp = self.last_result

        topk = max(math.ceil(top_ratio * graph.num_edges), 1)
        if topk == graph.num_edges:
            return graph

        exp_subgraph = copy.deepcopy(graph)
        exp_subgraph.y = graph.y

        order = list(np.argsort(-imp))
        topk_idx = order[:topk]
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[topk_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, topk_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x

        sub_nodes = torch.unique(exp_subgraph.edge_index)
        # .... the node features.
        exp_subgraph.x = graph.x[sub_nodes]

        exp_subgraph.batch = graph.batch[sub_nodes]
        row, col = graph.edge_index
        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((graph.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        exp_subgraph.edge_index = node_idx[exp_subgraph.edge_index]

        return exp_subgraph


    def evaluate_acc(self, top_ratio_list):

        assert self.last_result is not None
        acc_list = []
        for idx, top_ratio in enumerate(top_ratio_list):
            exp_subgraph = self.pack_explanatory_subgraph(top_ratio)
            self.model(exp_subgraph.x,
                       exp_subgraph.edge_index,
                       exp_subgraph.edge_attr,
                       exp_subgraph.batch
                       )
            acc_list.append(1 if exp_subgraph.y == self.model.readout.argmax(dim=1) else 0)
            #print("softmax: ", self.model.readout[0, exp_subgraph.y])

        return acc_list

    def evaluate_contrastivity(self):

        assert self.last_result is not None
        graph, imp = self.last_result
        imp += 1e-8
        imp = imp / imp.sum()
        counter_graph = copy.deepcopy(graph)
        counter_classes = [i for i in range(n_class_dict[self.model_name])]
        counter_classes.pop(graph.y)
        counter_accumulate = 0
        for c in counter_classes:
            counter_graph.y = torch.LongTensor([c]).cuda()
            if self.name == "Screener" and \
                    isinstance(graph.name[0], str) and \
                    "reddit" in graph.name[0]:
                counter_imp,_ = self.explain_graph(counter_graph, large_scale=True)
            else:
                counter_imp,_ = self.explain_graph(counter_graph)
            counter_imp += 1e-8
            counter_imp = counter_imp / counter_imp.sum()
            tmp = scipy.stats.spearmanr(counter_imp, imp)[0]

            if np.isnan(tmp):
                tmp = 1
            counter_accumulate += abs(tmp)
        self.last_result = graph, imp # may be unnecessary

        return counter_accumulate / len(counter_classes)

    # Monte Carlo simulation for calculating expected saliency change after
    # large perturbation in input  nodes features
    #def evaluate_validity(self, N=10):

    #    assert self.last_result is not None
    #    graph, imp = self.last_result

    #    noisy_graph = copy.deepcopy(graph)
    #    rank = self.get_rank(imp)
    #    EPS = float(val_pert * graph.x.mean())
    #    norm = len(rank) * graph.num_edges
    #    diff = []
    #    for _ in range(N):

    #        sample = torch.Tensor(graph.x.shape).uniform_(-3 * EPS, -EPS).cuda() if N % 2 else \
    #            torch.Tensor(graph.x.shape).uniform_(EPS, 3 * EPS).cuda()

    #        noisy_graph.x = graph.x + sample
    #        if self.name == "Screener" and isinstance(graph.name[0], str) and "reddit" in graph.name[0]:
    #            noisy_edge_imp = self.explain_graph(noisy_graph, large_scale = True)

    #       else:
    #            noisy_edge_imp = self.explain_graph(noisy_graph)
    #        noisy_rank = self.get_rank(noisy_edge_imp)
    #        diff.append(sum(abs(noisy_rank - rank)) / norm)

    #    self.last_result = graph, imp # may be unnecessary
    #    return torch.mean(torch.FloatTensor(diff))

    def evaluate_infidelity(self, N=5, p0 = 0.25):

        assert self.last_result is not None
        graph, imp = self.last_result

        imp = torch.FloatTensor(imp + 1e-8).cuda()
        imp = imp / imp.sum()
        ps = p0 * torch.ones_like(imp)

        self.model(graph.x,
                   graph.edge_index,
                   graph.edge_attr,
                   graph.batch
                   )
        ori_pred = self.model.readout[0, graph.y]
        lst = []
        for _ in range(N):

            p0 = torch.bernoulli(ps)
            edge_mask = (1.0 - p0).bool()
            self.model(graph.x,
                       graph.edge_index[:, edge_mask],
                       graph.edge_attr[edge_mask],
                       graph.batch
                       )
            pert_pred = self.model.readout[0, graph.y]
            infd = pow(sum(p0 * imp) - (ori_pred - pert_pred), 2).cpu().detach().numpy()
            lst.append(infd)
        lst = np.array(lst)
        return lst.mean()

    #def evaluate_infidelity(self, N=10, multipiler = 100):

    #    assert self.last_result is not None
    #    graph, imp = self.last_result
    #    imp = torch.FloatTensor(imp + 1e-16).unsqueeze(dim=0).cuda()
    #    imp = imp / imp.sum()
    #    EPS = float(ifd_pert * graph.edge_attr.mean())
    #    noisy_graph = copy.deepcopy(graph)

    #    self.model(graph.x,
    #              graph.edge_index,
    #               graph.edge_attr,
    #               graph.batch
    #               )
    #    ori_pred = self.model.readout[0, graph.y]

    #    lst = []
    #    for _ in range(N):

    #        sample = torch.Tensor(graph.edge_attr.shape).normal_(0, EPS).cuda()
    #        noisy_graph.edge_attr = graph.edge_attr - sample

    #        self.model(noisy_graph.x,
    #                   noisy_graph.edge_index,
    #                   noisy_graph.edge_attr,
    #                   noisy_graph.batch
    #                   )
    #        noisy_pred = self.model.readout[0, graph.y]
    #        ifd = pow((sample * imp.T).sum() - (ori_pred-noisy_pred), 2)
    #        lst.append(ifd)

    #    return torch.FloatTensor(lst).mean() * multipiler

    # evaluates the correlation between the importance assigned by the interpretability algorithm
    # to edges and the effect of each edge on the performance of the predictive model.
    #def evaluate_faithfulness(self, top_ratio_list):

    #    assert self.last_result is not None
    #    fai_list = []

    #    graph, imp = self.last_result
    #    cxplain_scores = self.get_cxplain_scores(graph)

    #    for idx, top_ratio in enumerate(top_ratio_list):

    #        topk = max(int(top_ratio * graph.num_edges), 1)
    #        topk_select = list(np.argsort(-imp))[:topk]

            # using the Spearmanr ranking correlation to measure the correlation between the edge importance and the performance
            # .... which accounts for the ranking of edge importance
            #tmp = np.corrcoef(imp[topk_select], cxplain_scores[topk_select])[0, 1]
    #        tmp = scipy.stats.spearmanr(imp[topk_select], cxplain_scores[topk_select])[0]
    #        if np.isnan(tmp):
    #            tmp = 1
    #        fai_list.append(tmp)
    #    return fai_list

    # def visualize(self, graph, edge_imp, method, vis_ratio=0.2, save=False):
    #
    #     topk = max(int(vis_ratio * graph.num_edges), 1)
    #
    #     if self.model_name == "VGNet":
    #         topk = 3
    #         idx = np.argsort(-edge_imp)[:topk]
    #         #idx = [25, 26]
    #         top_edges= graph.edge_index[:, idx]
    #         all =  graph.edge_index
    #         #nodes_idx = torch.unique(top_edges)
    #
    #         scene_graph = vgl.get_scene_graph(image_id=int(graph.name),
    #                                           images='module/visual_genome/raw',
    #                                           image_data_dir='module/visual_genome/raw/by-id/',
    #                                           synset_file='module/visual_genome/raw/synsets.json')
    #         #print([str(o) for o in scene_graph.objects].index('standing'))
    #         #print([str(o) for o in scene_graph.objects].index('surfboard'))
    #         #print([str(o) for o in scene_graph.objects].index('waves'))
    #         #print([str(o) for o in scene_graph.objects].index('shorts'))
    #         #top_edges = np.array([[26, 0, 8],
    #         #                      [7, 11,0]],dtype=np.int)
    #         # you can also use api to get the scence graph if the network is stable,
    #         # in this case, all the .json data is unneccessary
    #         # scene_graph = api.get_scene_graph_of_image(id=int(graph.id))
    #         '''
    #         T2 = ['standing', 'surfboard','man','waves','shorts']
    #         T1 = []
    #         T = T2
    #         '''
    #         print(scene_graph.relationships)
    #         for idx, e in enumerate(all.T):
    #             print(idx, '  ', scene_graph.objects[e[0]], '---', scene_graph.objects[e[1]])
    #         print(idx)
    #         r = 0.95 # transparency
    #         img = Image.open(r"data\VG\raw\%d-%d.jpg" % (graph.name, graph.y))
    #         data = list(img.getdata())
    #         ndata = list([(int((255-p[0])*r+p[0]),int((255-p[1])*r+p[1]),int((255-p[2])*r+p[2])) for p in data])
    #         mode = img.mode
    #         width, height = img.size
    #         '''
    #         r=0.9
    #         for t in T1:
    #             for n in nodes_idx:
    #                 obj = scene_graph.objects[n]
    #                 if not (str(obj) == t):
    #                     continue
    #                 for x in range(obj.x, obj.width+obj.x):
    #                     for y in range(obj.y, obj.y+obj.height):
    #                         ndata[y*width+x] = (int((255-data[y*width+x][0])*r+data[y*width+x][0]),
    #                                             int((255-data[y*width+x][1])*r+data[y*width+x][1]),
    #                                             int((255-data[y*width+x][2])*r+data[y*width+x][2]))
    #                 break
    #         '''
    #         edges = list(top_edges.T)
    #         print("prediction\n")
    #         for e in top_edges.T:
    #             print(scene_graph.objects[e[0]], '---', scene_graph.objects[e[1]])
    #         for i, (u, v) in enumerate(edges[::-1]):
    #             r = 1.0 - 1.0 / len(edges) * (i+1)
    #             #r=0
    #             obj1 = scene_graph.objects[u]
    #             obj2 = scene_graph.objects[v]
    #             for obj in [obj1, obj2]:
    #                 for x in range(obj.x, obj.width+obj.x):
    #                     for y in range(obj.y, obj.y+obj.height):
    #                         ndata[y * width + x] = (int((255 - data[y * width + x][0]) * r + data[y * width + x][0]),
    #                                                 int((255 - data[y * width + x][1]) * r + data[y * width + x][1]),
    #                                                 int((255 - data[y * width + x][2]) * r + data[y * width + x][2]))
    #
    #         img = Image.new(mode, (width, height))
    #         img.putdata(ndata)
    #
    #         plt.imshow(img)
    #         ax = plt.gca()
    #         for i, (u, v) in enumerate(edges):
    #             obj1 = scene_graph.objects[u]
    #             obj2 = scene_graph.objects[v]
    #             ax.annotate("", xy=(obj2.x, obj2.y), xytext=(obj1.x, obj1.y),
    #                         arrowprops=dict(width=topk-i, color='wheat', headwidth=5))
    #             for obj in [obj1, obj2]:
    #                 ax.text(obj.x, obj.y - 8, str(obj), style='italic',
    #                         fontsize=13,
    #                         bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 3,
    #                               'edgecolor': rec_color[i % len(rec_color)]}
    #                         )
    #                 ax.add_patch(Rectangle((obj.x, obj.y),
    #                                        obj.width,
    #                                        obj.height,
    #                                        fill=False,
    #                                        edgecolor=rec_color[i % len(rec_color)],
    #                                        linewidth=1.5))
    #         plt.tick_params(labelbottom='off', labelleft='off')
    #         plt.axis('off')
    #     else:
    #         idx = np.argsort(-edge_imp)[:topk]
    #
    #         G = nx.DiGraph()
    #         G.add_nodes_from(range(graph.num_nodes))
    #         G.add_edges_from(list(graph.edge_index.cpu().numpy().T))
    #         if self.vis_dict is None:
    #             self.vis_dict = vis_dict[self.model_name] if self.model_name in vis_dict.keys() else vis_dict['defult']
    #
    #         if graph.pos is None:
    #             graph.pos = nx.kamada_kawai_layout(G)
    #
    #         edge_pos_mask = np.zeros(graph.num_edges, dtype=np.bool_)
    #         edge_pos_mask[idx] = True
    #         vmax = 1#sum(edge_pos_mask)
    #         node_pos_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    #         node_neg_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    #         node_pos_idx = np.unique(graph.edge_index[:, edge_pos_mask].cpu().numpy()).tolist()
    #         node_neg_idx = list(set([i for i in range(graph.num_nodes)]) - set(node_pos_idx))
    #         node_pos_mask[node_pos_idx] = True
    #         node_neg_mask[node_neg_idx] = True
    #         labels = graph.z[0]
    #         label_dict = chem_graph_label_dict[self.model_name]
    #
    #         nx.draw_networkx_nodes(G, pos={i:graph.pos[i] for i in node_pos_idx},
    #                                nodelist=node_pos_idx,
    #                                node_size=self.vis_dict['node_size'],
    #                                node_color=graph.z[0][node_pos_idx],
    #                                alpha=1, cmap='winter',
    #                                linewidths=self.vis_dict['linewidths'],
    #                                edgecolors='red',
    #                                vmin=-max(graph.z[0]), vmax=max(graph.z[0])
    #                                )
    #         nx.draw_networkx_nodes(G, pos={i:graph.pos[i] for i in node_neg_idx},
    #                                nodelist=node_neg_idx,
    #                                node_size=self.vis_dict['node_size'],
    #                                node_color=graph.z[0][node_neg_idx],
    #                                alpha=0.1, cmap='winter',
    #                                linewidths=self.vis_dict['linewidths'],
    #                                edgecolors='whitesmoke',
    #                                vmin=-max(graph.z[0]), vmax=max(graph.z[0])
    #                                )
    #         nx.draw_networkx_labels(G, pos=graph.pos,
    #                                 labels={i: label_dict[labels[i]] for i in range(graph.num_nodes)},
    #                                 font_size=self.vis_dict['font_size'],
    #                                 font_weight='bold', font_color='k'
    #                                 )
    #         nx.draw_networkx_edges(G, pos=graph.pos,
    #                                edgelist=list(graph.edge_index.cpu().numpy().T),
    #                                edge_color='whitesmoke',
    #                                width=self.vis_dict['width'],
    #                                arrows=False
    #                                )
    #         nx.draw_networkx_edges(G, pos=graph.pos,
    #                                edgelist=list(graph.edge_index[:, edge_pos_mask].cpu().numpy().T),
    #                                edge_color=self.get_rank(edge_imp[edge_pos_mask]),
    #                                width=self.vis_dict['width'],
    #                                edge_cmap=cm.get_cmap('bwr'),
    #                                edge_vmin=-vmax, edge_vmax=vmax,
    #                                arrows=False
    #                                #arrows=True, arrowsize=3
    #                                )
    #         ax = plt.gca()
    #         ax.set_facecolor('aliceblue')
    #     if save:
    #         if method in ["Screener", "RandomCaster"]:
    #             folder = Path(r'image/%s/%s/r-%.2f' % (self.model_name, method, self.ratio))
    #         else:
    #             folder = Path(r'image/%s/%s' % (self.model_name, method))
    #         if not os.path.exists(folder):
    #             os.makedirs(folder)
    #         if isinstance(graph.name[0], str):
    #             plt.savefig(folder / Path(r'%d-%s.png' % (int(graph.y), str(graph.name[0]))), dpi=500, bbox_inches='tight')
    #         else:
    #             plt.savefig(folder / Path(r'%d-%d.png' % (graph.y, int(graph.name[0]))), dpi=500, bbox_inches='tight')
    #
    #     plt.cla()
