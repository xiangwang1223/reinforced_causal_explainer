from module.data_loader_zoo.mutag_dataloader import Mutagenicity

from torch_geometric.data import DataLoader

from module.utils import *
from module.utils.reorganizer import filter_correct_data

from rc_explainer_pool import RC_Explainer_pro
from backup.backup4.train_test_pool import train_policy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    set_seed(19930819)

    dataset_name = 'mutag'
    _hidden_size = 32
    _num_labels = 2
    debias_flag = False
    topN = None

    path = 'params/%s_net.pt' % dataset_name
    train_dataset = Mutagenicity('Data/MUTAG', mode='training')
    test_dataset = Mutagenicity('Data/MUTAG', mode='testing')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    model = torch.load(path)
    model.eval()

    model_1 = torch.load(path)
    model_1.eval()

    # refine the datasets and data loaders
    train_dataset, train_loader = filter_correct_data(model, train_dataset, train_loader, 'training')
    test_dataset, test_loader = filter_correct_data(model, test_dataset, test_loader, 'testing')

    rc_explainer = RC_Explainer_pro(_model=model_1, _num_labels=_num_labels,
                                    _hidden_size=_hidden_size, _use_edge_attr=True).to(device)
    pro_flag = False
    optimizer = rc_explainer.get_optimizer()

    topK_ratio = 0.1
    train_policy(rc_explainer, model, train_loader, test_loader, optimizer, topK_ratio,
                 debias_flag=debias_flag, topN=topN)