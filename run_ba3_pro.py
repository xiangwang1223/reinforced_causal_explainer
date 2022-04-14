from module.gnn_model_zoo.mutag_gnn import MutagNet
from module.data_loader_zoo.mutag_dataloader import Mutagenicity

from module.gnn_model_zoo.ba3motif_gnn import BA3MotifNet
from module.data_loader_zoo.ba3motif_dataloader import BA3Motif

from module.gnn_model_zoo.vg_gnn import VGNet
from module.data_loader_zoo.vg_dataloader import Visual_Genome

from torch_geometric.data import DataLoader

from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data, filter_correct_data_batch

from rc_explainer_pool import RC_Explainer, RC_Explainer_pro, RC_Explainer_Batch, \
    RC_Explainer_Batch_pro, RC_Explainer_Batch_pro2, RC_Explainer_Batch_pro3, RC_Explainer_Batch_star
from train_test_pool_batch2 import train_policy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_optimizer(rc_explainer, pro_flag=False):
    params = list(rc_explainer.edge_action_rep_generator.parameters())
    if pro_flag:
        for i_explainer in rc_explainer.edge_action_prob_generator:
            params += list(i_explainer.parameters())
    else:
        params += list(rc_explainer.edge_action_prob_generator.parameters())

    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=1e-5)
    return optimizer


if __name__ == '__main__':
    set_seed(19930819)

    dataset_name = 'ba3'
    _hidden_size = 64
    _num_labels = 3
    debias_flag = True
    topN = 5
    batch_size = 128

    path = 'params/%s_net.pt' % dataset_name
    train_dataset = BA3Motif('Data/BA3', mode='training')
    test_dataset = BA3Motif('Data/BA3', mode='testing')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    model = torch.load(path)
    model.eval()

    model_1 = torch.load(path)
    model_1.eval()

    # refine the datasets and data loaders
    train_dataset, train_loader = filter_correct_data_batch(model, train_dataset, train_loader, 'training', batch_size=batch_size)
    test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader, 'testing', batch_size=1)

    rc_explainer = RC_Explainer_Batch_star(_model=model_1, _num_labels=_num_labels,
                                          _hidden_size=_hidden_size, _use_edge_attr=False).to(device)
    pro_flag = False
    optimizer = rc_explainer.get_optimizer()

    topK_ratio = 0.2
    train_policy(rc_explainer, model, train_loader, test_loader, optimizer, topK_ratio,
                 debias_flag=debias_flag, topN=topN, batch_size=batch_size)