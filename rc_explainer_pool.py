from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
import torch.nn.functional as F
from module.utils import *
from torch_geometric.utils import softmax
from torch_scatter import scatter_max
from module.utils.reorganizer import relabel_graph, filter_correct_data
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RC_Explainer(torch.nn.Module):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer, self).__init__()

        self.model = _model
        self.model = self.model.to(device)
        self.model.eval()

        self.num_labels = _num_labels
        self.hidden_size = _hidden_size
        self.use_edge_attr = _use_edge_attr

        self.temperature = 0.1

        self.edge_action_rep_generator = Sequential(
            Linear(self.hidden_size * 2, self.hidden_size * 2),
            ReLU(),
            Linear(self.hidden_size * 2, self.hidden_size),
        ).to(device)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = Sequential(
            Linear(self.hidden_size * (1 + self.use_edge_attr), self.hidden_size),
            ReLU(),
            Linear(self.hidden_size, self.num_labels)
        ).to(device)
        return edge_action_prob_generator

    def forward(self, graph, state, target_y):
        ocp_edge_index = graph.edge_index.T[state].T
        ocp_edge_attr = graph.edge_attr[state]

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state]

        ava_node_reps_0 = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        ava_node_reps_1 = self.model.get_node_reps(graph.x, ocp_edge_index, ocp_edge_attr, graph.batch)
        ava_node_reps = ava_node_reps_0 - ava_node_reps_1

        ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                     ava_node_reps[ava_edge_index[1]]], dim=1).to(device)
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_action_reps, ava_edge_reps], dim=1)

        ava_action_probs = self.predict(ava_action_reps, target_y)

        return ava_action_probs

    def predict(self, ava_action_reps, target_y):
        action_probs = self.edge_action_prob_generator(ava_action_reps)[:, target_y]
        action_probs = action_probs.reshape(-1)

        action_probs = F.softmax(action_probs)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5):
        params = list(self.edge_action_rep_generator.parameters()) + \
                 list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer


class RC_Explainer_pro(RC_Explainer):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer_pro, self).__init__(_model, _num_labels, _hidden_size, _use_edge_attr=False)

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = nn.ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)).to(device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator

    def predict(self, ava_action_reps, target_y):
        i_explainer = self.edge_action_prob_generator[target_y]

        action_probs = i_explainer(ava_action_reps)
        action_probs = action_probs.reshape(-1)

        action_probs = F.softmax(action_probs)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5):
        params = list(self.edge_action_rep_generator.parameters())

        for i_explainer in self.edge_action_prob_generator:
            params += list(i_explainer.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer

"""
Using the mini-batch training the RC-Explainer, which is much efficient than the one-by-one training.
"""

class RC_Explainer_Batch(torch.nn.Module):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer_Batch, self).__init__()

        self.model = _model
        self.model = self.model.to(device)
        self.model.eval()

        self.num_labels = _num_labels
        self.hidden_size = _hidden_size
        self.use_edge_attr = _use_edge_attr

        self.temperature = 0.1

        self.edge_action_rep_generator = Sequential(
            Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 4),
            ELU(),
            Linear(self.hidden_size * 4, self.hidden_size * 2),
            ELU(),
            Linear(self.hidden_size * 2, self.hidden_size)
        ).to(device)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = Sequential(
            Linear(self.hidden_size, self.hidden_size),
            ELU(),
            Linear(self.hidden_size, self.num_labels)
        ).to(device)
        return edge_action_prob_generator

    def forward(self, graph, state, train_flag=False):
        ocp_edge_index = graph.edge_index.T[state].T
        ocp_edge_attr = graph.edge_attr[state]

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state]

        ava_node_reps_0 = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        ava_node_reps_1 = self.model.get_node_reps(graph.x, ocp_edge_index, ocp_edge_attr, graph.batch)
        ava_node_reps = ava_node_reps_0 - ava_node_reps_1

        ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                     ava_node_reps[ava_edge_index[1]]], dim=1).to(device)
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        ava_action_batch = graph.batch[ava_edge_index[0]]
        ava_y_batch = graph.y[ava_action_batch]

        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_action_reps, ava_edge_reps], dim=1)

        ava_action_probs = self.predict(ava_action_reps, ava_y_batch, ava_action_batch)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions

        return ava_action_probs, added_action_probs, added_actions

    def predict(self, ava_action_reps, target_y, ava_action_batch):
        action_probs = self.edge_action_prob_generator(ava_action_reps)
        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        action_probs = softmax(action_probs, ava_action_batch)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5, scope='all'):
        if scope in ['all']:
            params = self.parameters()
        else:
            params = list(self.edge_action_rep_generator.parameters()) + \
                     list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer

    def load_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = osp.join(osp.dirname(__file__), '..', '..', 'params', name)
        self.load_state_dict(torch.load(path))

    def save_policy_net(self, name='policy.pt', path=None):
        if not path:
            path = osp.join(osp.dirname(__file__), '..', '..', 'params', name)
        torch.save(self.state_dict(), path)


class RC_Explainer_Batch_star(RC_Explainer_Batch):
    def __init__(self, _model, _num_labels, _hidden_size, _use_edge_attr=False):
        super(RC_Explainer_Batch_star, self).__init__(_model, _num_labels, _hidden_size, _use_edge_attr=False)

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = nn.ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 2),
                ELU(),
                Linear(self.hidden_size * 2, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)
            ).to(device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator

    def forward(self, graph, state, train_flag=False):
        graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        if len(torch.where(state==True)[0]) == 0:
            subgraph_rep = torch.zeros(graph_rep.size()).to(device)
        else:
            subgraph = relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state]
        ava_node_reps = self.model.get_node_reps(graph.x, ava_edge_index, ava_edge_attr, graph.batch)

        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                         ava_node_reps[ava_edge_index[1]],
                                         ava_edge_reps], dim=1).to(device)
        else:

            ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                         ava_node_reps[ava_edge_index[1]]], dim=1).to(device)

        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        ava_action_batch = graph.batch[ava_edge_index[0]]
        ava_y_batch = graph.y[ava_action_batch]

        # get the unique elements in batch, in cases where some batches are out of actions.
        unique_batch, ava_action_batch = torch.unique(ava_action_batch, return_inverse=True)

        ava_action_probs = self.predict_star(graph_rep, subgraph_rep, ava_action_reps, ava_y_batch, ava_action_batch)

        # assert len(ava_action_probs) == sum(~state)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions, unique_batch

        return ava_action_probs, added_action_probs, added_actions, unique_batch

    def predict_star(self, graph_rep, subgraph_rep, ava_action_reps, target_y, ava_action_batch):
        action_graph_reps = graph_rep - subgraph_rep
        action_graph_reps = action_graph_reps[ava_action_batch]
        action_graph_reps = torch.cat([ava_action_reps, action_graph_reps], dim=1)

        action_probs = []
        for i_explainer in self.edge_action_prob_generator:
            i_action_probs = i_explainer(action_graph_reps)
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=1)

        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        # action_probs = softmax(action_probs, ava_action_batch)
        # action_probs = F.sigmoid(action_probs)
        return action_probs