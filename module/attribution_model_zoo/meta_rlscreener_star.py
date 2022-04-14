from math import sqrt
import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from torch.autograd import Variable
from itertools import count

import numpy as np

from module.utils import *
from module.utils.parser import *
# from module.utils.reorganizer import relabel_graph

from module.gnn_model_zoo.mutag_gnn import MutagNet
from module.data_loader_zoo.mutag_dataloader import Mutagenicity
from torch_geometric.data import DataLoader

from tqdm import tqdm
import copy

EPS = 1e-15
NegINF = -1e10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def filter_correct_data(model, dataset, loader, flag='Training'):
    graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
    idx = 0
    for g in tqdm(iter(loader), total=len(loader)):
        g.to(device)
        model(g.x, g.edge_index, g.edge_attr, g.batch)
        if g.y == model.readout.argmax(dim=1):
            graph_mask[idx] = True
        idx += 1

    loader = DataLoader(dataset[graph_mask], batch_size=1, shuffle=False)
    print("number of graphs in the %s:%4d" % (flag, graph_mask.nonzero().size(0)))
    return dataset, loader


def relabel_graph(graph, selection):
    subgraph = copy.deepcopy(graph)

    # retrieval properties of the explanatory subgraph
    # .... the edge_index.
    subgraph.edge_index = graph.edge_index.T[selection].T
    # .... the edge_attr.
    subgraph.edge_attr = graph.edge_attr[selection]
    # .... the nodes.
    sub_nodes = torch.unique(subgraph.edge_index)
    # .... the node features.
    subgraph.x = graph.x[sub_nodes]
    subgraph.batch = graph.batch[sub_nodes]

    row, col = graph.edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((graph.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    subgraph.edge_index = node_idx[subgraph.edge_index]

    return subgraph


class MetaRLScreener_pro(torch.nn.Module):

    def __init__(self, model, epochs=100, lr=0.01, log=True):
        super(MetaRLScreener_pro, self).__init__()
        self.model = model
        self.model.to(device)

        self.epochs = epochs
        self.lr = lr
        self.log = log

        self.temperature = 0.5

        self.edge_action_rep_generator = Sequential(
            Linear(32 * 3, 32),
            ELU(),
        )

        self.edge_action_prob_generator = Sequential(
            Linear(32, 32),
            ReLU(),
            Linear(32, 1)
        )

        #         self.edge_action_rep_generator = Sequential(
        #             Linear(32 * 3, 32),
        #             ELU(),
        #         )
        self.edge_action_rep_generator.to(device)
        self.edge_action_prob_generator.to(device)

    def forward(self, graph, selection):
        # 1. Get the subgraph per state based on the selection on the full graph.
        subgraph = relabel_graph(graph, selection)

        # 2. Get all node representations & all edge representations from the target model.
        node_reps = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        edge_reps = self.model.edge_emb(graph.edge_attr)

        # ---------------- Prepare materials for representing the state -----------------
        # 3. Get the representations of the full graph & subgraph
        graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        if len(torch.where(selection == True)[0]) == 0:
            subgraph_rep = 0.
        #             subgraph_rep = torch.zeros(graph_rep.shape)
        else:
            subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        # ---------------- Prepare materials for representing the action space -----------------
        # 4. Get the representations of the edge actions
        tmp_vec = torch.cat([node_reps[graph.edge_index[0]],
                             node_reps[graph.edge_index[1]],
                             edge_reps], dim=1)

        edge_action_reps = self.edge_action_rep_generator(tmp_vec.to(device))

        # ---------------- Perform the probability estimation of actions -----------------
        # 5. Get the probability of each edge being selected next.
        action_prob = self.estimate_edge_selection_prob(graph_rep, subgraph_rep, edge_action_reps, selection)

        #         n_graph_rep = subgraph_rep.repeat(edge_reps.shape[0], 1).cuda()
        #         tmp_vec = torch.cat([node_reps[graph.edge_index[0]],
        #                              node_reps[graph.edge_index[1]],
        #                              edge_reps,
        #                              n_graph_rep], dim=1)
        #         action_prob = self.edge_action_rep_generator(tmp_vec.to(device))
        #         action_prob[selection] = NegINF

        #         action_prob = F.softmax(action_prob/self.temperature)

        return action_prob

    def estimate_edge_selection_prob(self, graph_rep, subgraph_rep, edge_action_reps, selection):
        graph_diff_rep = graph_rep - subgraph_rep
        graph_diff_rep = graph_diff_rep.reshape(-1, 1)

        action_prob = torch.matmul(edge_action_reps, graph_diff_rep)
        action_prob = action_prob.reshape(-1)

        action_prob[selection] = NegINF

        action_prob = F.softmax(action_prob / self.temperature)

        return action_prob

    def estimate_edge_selection_prob_2(self, graph_rep, subgraph_rep, edge_action_reps, selection):
        graph_diff_rep = (graph_rep - subgraph_rep)

        action_prob = self.edge_action_prob_generator(graph_diff_rep * edge_action_reps)
        action_prob = action_prob.reshape(-1)

        action_prob[selection] = NegINF

        action_prob = F.softmax(action_prob / self.temperature)

        return action_prob


if __name__ == '__main__':
    set_seed(19930819)

    dataset_name = 'mutag'
    path = 'params/%s_net.pt' % dataset_name
    train_dataset = Mutagenicity('Data/MUTAG', mode='training')
    test_dataset = Mutagenicity('Data/MUTAG', mode='testing')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MutagNet().to(device)
    model = torch.load(path)
    model.eval()

    train_dataset, train_loader = filter_correct_data(model, train_dataset, train_loader, 'training')
    test_dataset, test_loader = filter_correct_data(model, test_dataset, test_loader, 'testing')

    num_episode = 100
    gamma = 0.99

    rl_screener = MetaRLScreener_pro(model=model)

    params = list(rl_screener.edge_action_rep_generator.parameters()) + list(
        rl_screener.edge_action_prob_generator.parameters())

    optimizer = torch.optim.Adam(params, lr=0.01)

    avg_reward = 0.
    criterion = nn.CrossEntropyLoss()

    topK_ratio = 0.1

    state_pool = []
    action_pool = []
    reward_pool = []

    for epoch in range(num_episode):
        steps = 0
        epoch_reward = 0.
        rl_screener.train()
        for graph in tqdm(iter(train_loader), total=len(train_loader)):
            graph.to(device)
            #         print(graph)

            max_budget = graph.num_edges
            # reset the state
            state = torch.zeros(max_budget, dtype=torch.bool)

            valid_budget = max(int(topK_ratio * max_budget), 1)
            for budget in range(valid_budget):
                # selection action with the probability
                action_prob = rl_screener(graph=graph, selection=state)
                #             print(action_prob)
                #             print('--- action_prob ----')

                m = Categorical(action_prob)
                action = m.sample()

                state_pool.append(state)
                action_pool.append(action)

                # generate next state & get reward
                state[action] = True
                subgraph = relabel_graph(graph, state)

                subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
                reward = (-1) * criterion(subgraph_pred, subgraph.y)

                #             reward_pool.append(reward)
                reward_pool.append(reward.cpu().detach().numpy())

                #             print(reward.cpu().detach().numpy())
                #             print('---- reward ----')

                steps += 1

            #         print(reward_pool)
            #         print('---- reward pool ----')

            # Update policy
            running_add = 0
            # Discount reward
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    #                 running_add = running_add * gamma + reward_pool[i]
                    #                 reward_pool[i] = running_add

                    reward_pool[i] = running_add * gamma + reward_pool[i]

            #                 running_add = running_add * gamma + reward_pool[i]
            #                 reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)

            #         for i in range(steps):
            #             reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Descent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = action_pool[i]
                reward = reward_pool[i]

                action_prob = rl_screener(graph=graph, selection=state)
                m = Categorical(action_prob)

                loss = - m.log_prob(action) * reward
                loss.backward()
            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

        # ------------- Testing ----------------------
        rl_screener.eval()
        acc_count = 0
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            graph.to(device)

            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)

            valid_budget = max(int(topK_ratio * max_budget), 1)
            for budget in range(valid_budget):
                action_prob = rl_screener(graph=graph, selection=state)

                make_action_id = torch.argmax(action_prob)
                make_action_prob = action_prob[make_action_id]

                state[make_action_id] = True

            subgraph = relabel_graph(graph, state)
            subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

            if graph.y == subgraph_pred.argmax(dim=1):
                acc_count += 1
        print('test acc: %.4f' % (acc_count / len(test_loader)))