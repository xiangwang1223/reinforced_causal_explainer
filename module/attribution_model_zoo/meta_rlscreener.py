from math import sqrt
import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU
import torch.nn.functional as F
from module.utils import *
from module.utils.parser import *
from module.utils.reorganizer import relabel_graph

from module.data_loader_zoo.mutag_dataloader import Mutagenicity
from torch_geometric.data import DataLoader

from tqdm import tqdm

EPS = 1e-15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaRLScreener(torch.nn.Module):

    def __init__(self, model, epochs=100, lr=0.01, log=True):
        super(MetaRLScreener, self).__init__()
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

    def explain_graph_per_state(self, graph, selection):
        # 1. Get the subgraph per state based on the selection on the full graph.
        subgraph = relabel_graph(graph, selection)

        # 2. Get all node representations & all edge representations from the target model.
        node_reps = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        edge_reps = self.model.edge_embed(graph.edge_attr)

        # ---------------- Prepare materials for representing the state -----------------
        # 3. Get the representations of the full graph & subgraph
        graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        if len(torch.where(selection == True)[0]) == 0:
            subgraph_rep = 0.
        else:
            subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        # ---------------- Prepare materials for representing the action space -----------------
        # 4. Get the representations of the edge actions
        edge_action_reps = self.edge_action_rep_generator(torch.cat([node_reps[graph.edge_index[0]],
                                                                     node_reps[graph.edge_index[1]],
                                                                     edge_reps], dim=1))

        # ---------------- Perform the probability estimation of actions -----------------
        # 5. Get the probability of each edge being selected next.
        edge_action_prob = self.__make_one_action__(graph_rep, subgraph_rep, edge_action_reps, selection)

        return edge_action_prob

    def __make_one_action__(self, graph_rep, subgraph_rep, edge_action_reps, selection):
        state_prob = torch.matmul(edge_action_reps, graph_rep - subgraph_rep)
        state_prob = state_prob.reshape(1, -1)

        min_prob = torch.min(state_prob)
        state_prob[selection] = min_prob

        return F.softmax(state_prob/self.temperature, dim=1)


def data_split(dataset, train_ratio=0.8):
    dataset_size = len(dataset)
    train_size = dataset_size * 0.8

    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('Split datasets & Generate dataloaders: DONE')

    return train_dataset, test_dataset, train_loader, test_loader


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


if __name__ == "__main__":

    args = parse_args()
    set_seed(args.random_seed)

    dataset_name = 'mutag'
    path = 'params/%s_net.pt' % dataset_name
    dataset = Mutagenicity('Data/MUTAG')

    train_dataset, test_dataset, train_loader, test_loader = data_split(dataset)

    model = torch.load(path).to(device)
    model.eval()

    train_dataset, train_loader = filter_correct_data(model, train_dataset, train_loader, 'training')
    test_dataset, test_loader = filter_correct_data(model, test_dataset, test_loader, 'testing')

    rl_screener = MetaRLScreener(model=model)
    optimizer = torch.optim.Adam(rl_screener.edge_action_rep_generator, lr=0.01)

    avg_reward = 0.
    criterion = nn.CrossEntropyLoss()

    topK_ratio = 0.1

    for epoch in range(100):
        epoch_reward = 0.
        rl_screener.train()
        for graph in tqdm(iter(train_loader), total=len(train_loader)):
            graph.to(device)

            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)

            for budget in range(max(topK_ratio * max_budget, 1)):
                edge_action_prob = rl_screener.explain_graph_per_state(graph=graph, selection=state)

                make_action_id = torch.argmax(edge_action_prob)
                make_action_prob = edge_action_prob[make_action_id]

                state[make_action_id] = True

                subgraph = relabel_graph(graph, state)
                subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
                reward = (-1) * criterion(subgraph_pred, subgraph.y)

                epoch_reward += reward
                reward -= avg_reward

                loss = (-1) * reward * torch.log(make_action_prob)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # ------------- Testing ----------------------
        rl_screener.eval()
        acc_count = 0
        for graph in tqdm(iter(test_loader), total=len(test_loader)):
            graph.to(device)

            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)

            for budget in range(max(topK_ratio * max_budget, 1)):
                edge_action_prob = rl_screener.explain_graph_per_state(graph=graph, selection=state)

                make_action_id = torch.argmax(edge_action_prob)
                make_action_prob = edge_action_prob[make_action_id]

                state[make_action_id] = True

            subgraph = relabel_graph(graph, state)
            subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

            if graph.y == subgraph_pred.argmax(dim=1):
                acc_count += 1
        print('test acc: %.4f' %(acc_count/len(test_loader)))