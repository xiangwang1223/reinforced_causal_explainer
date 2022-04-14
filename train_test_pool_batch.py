import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data

from tqdm import tqdm
from torch_scatter import scatter_max

EPS = 1e-15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_policy_all_with_gnd(rc_explainer, model, test_loader, topN=None):
    rc_explainer.eval()
    model.eval()

    topK_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_count_list = np.zeros(len(topK_ratio_list))

    precision_topN_count = 0.
    recall_topN_count = 0.

    with torch.no_grad():
        for graph in iter(test_loader):
            graph = graph.to(device)
            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)

            check_budget_list = [max(int(_topK * max_budget), 1) for _topK in topK_ratio_list]
            valid_budget = max(int(0.9 * max_budget), 1)

            for budget in range(valid_budget):
                available_actions = state[~state].clone()

                _, _, make_action_id = rc_explainer(graph=graph, state=state, train_flag=False)

                available_actions[make_action_id] = True
                state[~state] = available_actions.clone()

                if (budget + 1) in check_budget_list:
                    check_idx = check_budget_list.index(budget + 1)
                    subgraph = relabel_graph(graph, state)
                    subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

                    acc_count_list[check_idx] += sum(graph.y == subgraph_pred.argmax(dim=1))

                if topN is not None and budget == topN - 1:
                    precision_topN_count += torch.sum(state*graph.ground_truth_mask[0])/topN
                    recall_topN_count += torch.sum(state*graph.ground_truth_mask[0])/sum(graph.ground_truth_mask[0])

    acc_count_list[-1] = len(test_loader)
    acc_count_list = np.array(acc_count_list)/len(test_loader)

    precision_topN_count = precision_topN_count / len(test_loader)
    recall_topN_count = recall_topN_count / len(test_loader)

    if topN is not None:
        print('ACC-AUC: %.4f, Precision@5: %.4f, Recall@5: %.4f\n' %
              (acc_count_list.mean(), precision_topN_count, recall_topN_count))
    else:
        print('ACC-AUC: %.4f\n' % acc_count_list.mean())
    print(acc_count_list)

    return np.array(acc_count_list)/len(test_loader)


def normalize_reward(reward_pool):
    reward_pool = torch.stack(reward_pool)

    if reward_pool.shape[0] != 1:
        reward_mean = torch.mean(reward_pool)
        reward_std = torch.std(reward_pool) + EPS
        reward_pool = (reward_pool - reward_mean) / reward_std
    return reward_pool


def bias_detector(model, graph, valid_budget):
    pred_bias_list = []

    for budget in range(valid_budget):
        num_repeat = 2

        i_pred_bias = 0.
        for i in range(num_repeat):
            bias_selection = torch.zeros(graph.num_edges, dtype=torch.bool)

            ava_action_batch = graph.batch[graph.edge_index[0]]
            ava_action_probs = torch.rand(ava_action_batch.size()).to(device)
            _, added_actions = scatter_max(ava_action_probs, ava_action_batch)

            bias_selection[added_actions] = True
            bias_subgraph = relabel_graph(graph, bias_selection)
            bias_subgraph_pred = model(bias_subgraph.x, bias_subgraph.edge_index,
                                       bias_subgraph.edge_attr, bias_subgraph.batch).detach()

            i_pred_bias += bias_subgraph_pred / num_repeat

        pred_bias_list.append(i_pred_bias)

    return pred_bias_list


def train_policy(rc_explainer, model, train_loader, test_loader, optimizer,
                 topK_ratio=0.1, debias_flag=False, topN=None, batch_size=32):
    num_episodes = 100

    test_policy_all_with_gnd(rc_explainer, model, test_loader, topN)
    ep = 0

    while ep < num_episodes:
        rc_explainer.train()
        model.eval()

        loss = 0.
        avg_reward = []

        for graph in tqdm(iter(train_loader), total=len(train_loader)):
            graph = graph.to(device)

            if topK_ratio < 1:
                valid_budget = max(int(topK_ratio * graph.num_edges / batch_size), 1)
            else:
                valid_budget = topK_ratio

            batch_loss = 0.

            full_subgraph_pred = F.softmax(model(graph.x, graph.edge_index,
                                                 graph.edge_attr, graph.batch)).detach()

            current_state = torch.zeros(graph.num_edges, dtype=torch.bool)

            if debias_flag:
                pred_bias_list = bias_detector(model, graph, valid_budget)

            pre_reward = 0.
            for budget in range(valid_budget):
                available_action = current_state[~current_state].clone()

                _, added_action_probs, added_actions = rc_explainer(graph, current_state, train_flag=True)

                new_state = current_state.clone()
                try:
                    available_action[added_actions] = True
                except:
                    pass

                new_state[~current_state] = available_action

                new_subgraph = relabel_graph(graph, new_state)
                new_subgraph_pred = model(new_subgraph.x, new_subgraph.edge_index,
                                          new_subgraph.edge_attr, new_subgraph.batch)

                if debias_flag:
                    new_subgraph_pred = F.softmax(new_subgraph_pred - pred_bias_list[budget]).detach()
                else:
                    new_subgraph_pred = F.softmax(new_subgraph_pred).detach()

                reward = get_reward(full_subgraph_pred, new_subgraph_pred, graph.y, mode='binary', pre_reward=pre_reward)
                pre_reward = reward

                batch_loss += torch.mean(- torch.log(added_action_probs + EPS) * reward)
                avg_reward += reward.tolist()

                current_state = new_state.clone()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        avg_reward = torch.mean(torch.FloatTensor(avg_reward))

        ep += 1
        print('Episode: %d, loss: %.4f, average rewards: %.4f' % (ep, loss.detach(), avg_reward.detach()))

        test_policy_all_with_gnd(rc_explainer, model, test_loader, topN)

        rc_explainer.train()
    return rc_explainer


def get_reward(full_subgraph_pred, new_subgraph_pred, target_y, mode='mutual_info', pre_reward=0.):
    if mode in ['mutual_info']:
        reward = torch.sum(full_subgraph_pred * torch.log(new_subgraph_pred + EPS), dim=1)
        reward += 2 * (target_y == new_subgraph_pred.argmax(dim=1)).float() - 1.

    elif mode in ['binary']:
        reward = (target_y == new_subgraph_pred.argmax(dim=1)).float()
        reward = 2. * reward - 1.

    elif mode in ['cross_entropy']:
        reward = torch.log(new_subgraph_pred + EPS)[:, target_y]

    # reward += pre_reward
    reward += 0.9 * pre_reward

    return reward