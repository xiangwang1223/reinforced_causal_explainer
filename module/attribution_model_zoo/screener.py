import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from torch_geometric.data import Data, Dataset, DataLoader
from module.attribution_model_zoo.basic_explainer import Explainer
from module.attribution_model_zoo.gnnexplainer import GNNExplainer
from module.attribution_model_zoo.sa_explainer import SAExplainer
from module.attribution_model_zoo.deeplift import DeepLIFTExplainer
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_sparse import SparseTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-4


class Screener(Explainer):

    n_max_candidates = 10

    def __init__(self, gnn_model_path):
        super(Screener, self).__init__(gnn_model_path=gnn_model_path)

    def sa_prior(self, graph):

        scores = SAExplainer(self.path).explain_graph(graph)
        return scores, np.argsort(-scores)

    def deeplift_prior(self, graph):

        scores = DeepLIFTExplainer(self.path).explain_graph(graph)
        return scores, np.argsort(-scores)

    def filter_edges(self, graph, edge_list):
        return (graph.edge_attr[edge_list], graph.edge_index[:, edge_list])

    @staticmethod
    def cosine_similarity(v_1, v_2):
        return torch.cosine_similarity(v_1, v_2, dim=0)

    @staticmethod
    def inner_product(v_1, v_2):
        return torch.dot(v_1, v_2)

    def diff_pool_cluster(self, graph, C, epoch=150, lr=0.1):

        N, E = graph.num_nodes, graph.num_edges
        x = graph.x
        adj = torch.sparse.FloatTensor(graph.edge_index,
                                       graph.edge_attr.sum(axis=1).view(-1)/2.,
                                       torch.Size([N, N])
                                       ).to_dense()
        s = torch.randn(size=(N, C), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([s], lr=lr)
        for _ in range(epoch):
            x_out, adj_out, l1, e1 = dense_diff_pool(x, adj, s)
            loss = l1 + e1
            loss.backward()
            optimizer.step()
        return s, x_out, adj_out

    def cluster_edge_index(self, edge_index, node_map):

        C = node_map.max() + 1

        edge_map = [[[] for _ in range(C)] for _ in range(C)]
        edge_index = node_map[edge_index].cpu().detach().numpy()
        new_edge_index = set()
        for idx, edge in enumerate(edge_index.T):
            edge_map[edge[0]][edge[1]].append(idx)
            new_edge_index.add((edge[0], edge[1]))

        return np.array(edge_map), torch.LongTensor(list(new_edge_index)).T

    def explain_graph(self, graph,
                      model=None,
                      ratio=1,
                      draw_graph=0,
                      vis_ratio=0.2,
                      C=10,
                      large_scale=False,
                      return_ice_ratio=0.):

        if model == None:
            model = self.model

        E = graph.num_edges
        # 1. if large_scale is True, downscale the graph first before assigning importance
        if large_scale:
            graph.x = torch.flatten(graph.x, start_dim=1, end_dim=-1)
            alpha, _, _ = self.diff_pool_cluster(graph, C=C)
            #alpha = torch.nn.functional.softmax(alpha, dim=1)
            node_cluster_map = alpha.argmax(dim=1)
            edge_cluster_map, graph_edge_index = self.cluster_edge_index(graph.edge_index, node_cluster_map)
            E = graph_edge_index.size(1)
            #print("number of edges: %d" % E)

        # 2. confirm the number of selected edges, or size of explanatory graph
        topk = max(int(ratio * E), 1)
        self.ratio = ratio

        # 3. get the ground truth labels (if only considering the correctly predicted graph instances)
        y = graph.y

        # 4. initialize the optimal edge set.
        opt_edges = []

        # 5. initialize the whole edge set:
        #    say, all_edges = [1,2,3,4,5], scores = [0,0,0,0,0]
        all_edges = [i for i in range(E)]
        scores = np.zeros(E)

        # 6. initialize the ranking list of edges via cxplain:
        #    say, initial_rank = [3,2,4,1,5]
        initial_score, initial_rank = self.sa_prior(graph)
        initial_rank = initial_rank.tolist()

        # 7. generate initial score under large-scale graph circumstance
        if large_scale:
            c_initial_score = np.zeros(E)
            for idx in range(E):
                c_1, c_2 = graph_edge_index[:, idx]
                c_initial_score[idx] = initial_score[edge_cluster_map[c_1,c_2]].sum()
            initial_score = c_initial_score
            initial_rank = np.argsort(-initial_score).tolist()
        model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        ori_pred = model.readout[0].detach()
        criterion = nn.CrossEntropyLoss()
        for k in range(topk):
            with torch.no_grad():

                # 8. generate the rest edge set (i.e., rest_edges) to check by removing these already selected:
                #    say, rest_edges = [1,2,4,5]
                rest_edges = list(set(all_edges) - set(opt_edges))
                n_rest_edges = len(rest_edges)

                min_loss = 1e6
                replace_cur = True

                if n_rest_edges < self.n_max_candidates:
                    candidates = rest_edges
                    # 9.  select the first entry of initial_rank:
                    #    say, cur_edge = 3
                    cur_edge = initial_rank[0]
                else:
                    p = initial_score[rest_edges] + EPS
                    p = p / p.sum()
                    candidates = np.random.choice(rest_edges,
                                                  self.n_max_candidates,
                                                  replace=False,
                                                  p=p)
                    # 9. select the first entry of candidates
                    cur_edge = candidates[0]

                candidates = list(candidates)
                # 10. get graph representation for each edge in rest edge set (manually create a batch)
                n_test = len(candidates)
                x = copy.copy(graph.x).to(device)

                x = x.repeat([n_test] + [1 for _ in range(x.dim() - 1)])
                edge_attr = torch.tensor([]).to(device)
                edge_index = torch.LongTensor([]).to(device)
                batch = torch.LongTensor([]).to(device)

                for idx, chk_edge in enumerate(candidates):

                    tmp_1 = opt_edges.copy()
                    tmp_1.append(chk_edge)
                    if large_scale:
                        row, col = graph_edge_index[:, tmp_1].numpy()
                        tmp_1 =  edge_cluster_map[row, col][0]

                    edge_attr_1, edge_index_1 = self.filter_edges(graph, tmp_1)

                    batch = torch.cat([batch, graph.batch + idx])
                    edge_attr = torch.cat([edge_attr, edge_attr_1], dim=0)
                    edge_index = torch.cat([edge_index, edge_index_1 + idx * graph.num_nodes], dim=1)

                rest_edges_g_rep = model.get_graph_rep(x, edge_index, edge_attr, batch)

            # 11. screening/checking all edges in the rest_edges
            for idx, chk_edge in enumerate(candidates):
                # 12. generate a temporary edge set (i.e., tmp) by adding the edge being pre-selected (i.e., cur_edge)
                #    into the optimal set (i.e., opt_edges)
                #    say, tmp = [3]
                if replace_cur:

                    g_rep_2 = rest_edges_g_rep[candidates.index(cur_edge)].view(1, -1)
                    # 13. obtain the gradient w.r.t. cur_edge & the prediction of tmp.
                    g_rep_2 = Variable(g_rep_2, requires_grad=True)
                    pred = model.get_pred(g_rep_2)
                    #model.get_pred(g_rep_2)
                    #pred = model.readout[0]
                    #loss = torch.norm(ori_pred[graph.y] - pred[graph.y])
                    #loss = torch.norm(ori_pred - pred)
                    #loss.backward()
                    loss = criterion(pred, graph.y)
                    loss.backward()

                    # 14. obtain the gradient representation of cur_edge
                    v_2 = g_rep_2.grad.view(-1)


                g_rep_1 = rest_edges_g_rep[idx]
                # 15. obtain the difference representations of cur_edge & chk_edge
                #     say, which is better, cur_edge (3) or chk_edge (1), to minimize the loss
                # v_1 = self.dif_vector(graph, chk_edge, cur_edge, op_type='concate')
                v_1 = (g_rep_1 - g_rep_2).view(-1)

                # 16. obtain the chk_loss, comparing with the minimized loss (i.e., min_loss)
                chk_loss = loss + self.inner_product(v_1, v_2)
                #print(entropy_loss)
                # 17. replace the cur_edge with the better chk_edge
                if chk_loss < min_loss:
                    replace_cur = True
                    cur_edge, min_loss = chk_edge, chk_loss
                else:
                    replace_cur = False

            # 18. add the best edge, cur_edge, into the opt_edges
            opt_edges.append(cur_edge)
            scores[cur_edge] = topk - k

            # 19. remove the best edge from the initial_rank.
            initial_rank.remove(cur_edge)
            #torch.cuda.empty_cache()

        # 20. remap the cluster score to node cluster importance and further compute
        # edge scores in the original graph
        if large_scale:
            beta = torch.sparse.FloatTensor(graph_edge_index,
                                            torch.FloatTensor(scores),
                                            torch.Size([C, C])
                                           ).to_dense().to(device)
            node_cluster_imp = torch.mm(alpha, beta)
            row, col = graph.edge_index
            edge_cluster_imp = torch.mul(node_cluster_imp[row], node_cluster_imp[col])
            scores = edge_cluster_imp.sum(dim=1).cpu().detach().numpy()

        scores = self.norm_imp(scores)

        self.last_result = (graph, scores)
        if draw_graph:
            self.visualize(graph, scores, 'Screener', vis_ratio=vis_ratio)

        ICEs = []
        # calculate the ICE
        if return_ice_ratio > 0:
            Ek = []
            n_test = min(max(int(0.2 * len(opt_edges)), 2), len(opt_edges))
            x = copy.copy(graph.x).to(device)

            x = x.repeat([n_test] + [1 for _ in range(x.dim() - 1)])
            edge_attr = torch.tensor([]).to(device)
            edge_index = torch.LongTensor([]).to(device)
            batch = torch.LongTensor([]).to(device)

            for idx, ek in enumerate(opt_edges[:n_test]):
                tmp = Ek.copy()
                Ek.append(ek)
                if large_scale:
                    row, col = graph_edge_index[:, Ek].numpy()
                    tmp = edge_cluster_map[row, col][0]
                edge_attr_1, edge_index_1 = self.filter_edges(graph, tmp)

                batch = torch.cat([batch, graph.batch + idx])
                edge_attr = torch.cat([edge_attr, edge_attr_1], dim=0)
                edge_index = torch.cat([edge_index, edge_index_1 + idx * graph.num_nodes], dim=1)

            preds = model(x, edge_index, edge_attr, batch)
            for k in range(len(preds) - 1):
                ICE = float(
                    criterion(preds[k].unsqueeze(0), graph.y) - float(criterion(preds[k + 1].unsqueeze(0), graph.y)))
                ICEs.append(ICE if ICE > 1e-5 else 0)

        return scores, ICEs