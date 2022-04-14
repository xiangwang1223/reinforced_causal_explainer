from module.gnn_model_zoo.mutag_gnn import MutagNet
from module.data_loader_zoo.mutag_dataloader import Mutagenicity

from module.gnn_model_zoo.ba3motif_gnn import BA3MotifNet
from module.data_loader_zoo.ba3motif_dataloader import BA3Motif

from module.gnn_model_zoo.vg_gnn import VGNet
from module.data_loader_zoo.vg_dataloader import Visual_Genome

from module.gnn_model_zoo.reddit5k_gnn import Reddit5kNet
from module.data_loader_zoo.reddit5k_dataloader import Reddit5k

from torch_geometric.data import DataLoader

from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data, filter_correct_data_batch
from module.utils.parser import parse_args
from module.utils.logging import Logger

from rc_explainer_pool import RC_Explainer, RC_Explainer_pro, RC_Explainer_Batch, RC_Explainer_Batch_star
from train_test_pool_batch3 import train_policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def configuration(dataset_name):
    configs = dict()
    if dataset_name in ['vg']:
        configs['_hidden_size'] = 128
        configs['_num_labels'] = 5
        configs['debias_flag'] = False
        configs['topN'] = None
        configs['batch_size'] = 64
        configs['scope'] = 'part'

        configs['train_dataset'] = Visual_Genome('Data/VG', mode='training')
        configs['test_dataset'] = Visual_Genome('Data/VG', mode='testing')

        configs['topK_ratio'] = 0.1

    elif dataset_name in ['ba3']:
        configs['_hidden_size'] = 64
        configs['_num_labels'] = 3
        configs['debias_flag'] = True
        configs['topN'] = 5
        configs['batch_size'] = 64
        configs['scope'] = 'part'

        configs['train_dataset'] = BA3Motif('Data/BA3', mode='training')
        configs['test_dataset'] = BA3Motif('Data/BA3', mode='testing')

        configs['topK_ratio'] = 10

    elif dataset_name in ['mutag']:
        configs['_hidden_size'] = 32
        configs['_num_labels'] = 2
        configs['debias_flag'] = False
        configs['topN'] = None
        configs['batch_size'] = 64
        configs['scope'] = 'all'

        configs['train_dataset'] = Mutagenicity('Data/MUTAG', mode='training')
        configs['test_dataset'] = Mutagenicity('Data/MUTAG', mode='testing')

        configs['topK_ratio'] = 0.1

    elif dataset_name in ['reddit5k']:
        configs['_hidden_size'] = 32
        configs['_num_labels'] = 5
        configs['debias_flag'] = False
        configs['topN'] = None
        configs['batch_size'] = 64
        configs['scope'] = 'part'

        configs['train_dataset'] = Reddit5k('Data/Reddit5k', mode='training')
        configs['test_dataset'] = Reddit5k('Data/Reddit5k', mode='testing')

        configs['topK_ratio'] = 0.1

    return configs


if __name__ == '__main__':
    set_seed(19930819)
    args = parse_args()

    if not torch.cuda.is_available():
        args.dataset_name = 'reddit5k'
        args.lr = 0.0001
        args.l2 = 0.0001
        args.reward_mode = 'mutual_info'

    dataset_name = args.dataset_name

    # get the configuration for a specific dataset
    configs = configuration(dataset_name)

    _hidden_size = configs['_hidden_size']
    _num_labels = configs['_num_labels']
    debias_flag = configs['debias_flag']
    topN = configs['topN']
    batch_size = configs['batch_size']
    scope = configs['scope']

    # get the trianing & testing datasets
    train_dataset = configs['train_dataset']
    test_dataset = configs['test_dataset']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    path = 'params/%s_net.pt' % dataset_name
    if torch.cuda.is_available():
        model = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
    else:
        model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()

    if torch.cuda.is_available():
        model_1 = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
    else:
        model_1 = torch.load(path, map_location=torch.device('cpu'))
    model_1.eval()

    # refine the datasets and data loaders
    train_dataset, train_loader = filter_correct_data_batch(model, train_dataset, train_loader, 'training',
                                                            batch_size=batch_size)
    test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader, 'testing',
                                                          batch_size=1)

    rc_explainer = RC_Explainer_Batch_star(_model=model_1, _num_labels=_num_labels,
                                           _hidden_size=_hidden_size, _use_edge_attr=False).to(device)

    lr = args.lr
    weight_decay = args.l2
    reward_mode = args.reward_mode

    optimizer = rc_explainer.get_optimizer(lr=lr, weight_decay=weight_decay, scope=scope)

    topK_ratio = configs['topK_ratio']

    save_model_path = 'explainer_params/%s_%s_%s_%s_new.pt' % (dataset_name, reward_mode, str(lr), str(weight_decay))
    rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec = \
        train_policy(rc_explainer, model, train_loader, test_loader, optimizer, topK_ratio,
                     debias_flag=debias_flag, topN=topN, batch_size=batch_size, reward_mode=reward_mode,
                     save_model_path=save_model_path)

    logger = open('explainer_output/%s_output_new.log' % dataset_name, 'a')
    logger.write('Reward-Mode: %s, lr: %s, l2: %s\n' % (reward_mode, str(lr), str(weight_decay)))
    logger.write('ACC-AUC: %.4f, P@5: %.4f, R@5: %.4f\n' % (best_acc_auc, best_pre, best_rec))
    logger.write('ACC-Curves: %s\n\n' % best_acc_curve)
    logger.close()
