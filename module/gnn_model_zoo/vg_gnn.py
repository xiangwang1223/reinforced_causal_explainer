import argparse
import copy
import os.path as osp
import time

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential as Seq, ReLU, Tanh, Linear as Lin, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from torch_geometric.nn import APPNP, BatchNorm, global_mean_pool, LEConv, SGConv
from module.utils import set_seed, Gtrain, Gtest
from module.data_loader_zoo.vg_dataloader import Visual_Genome
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual Genome Model")

    parser.add_argument('--data_path', nargs='?',
                        default=osp.join(osp.dirname(__file__), '..', '..', 'Data', 'VG'),
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?',
                        default=osp.join(osp.dirname(__file__), '..', '..', 'params'),
                        help='path for saving trained model.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=0.5 * 1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--random_label', type=bool, default=False,
                        help='train a model under label randomization for sanity check')

    return parser.parse_args()

class VGNet(torch.nn.Module):

    def __init__(self):
        super(VGNet, self).__init__()

        self.node_encode = nn.Sequential(OrderedDict([
                # With square kernels and equal stride
                ('conv1', nn.Conv2d(3, 8, 3, stride=1)),
                ('relu1', nn.ReLU()),
                ('conv2',nn.Conv2d(8, 5, 5, stride=2)),
                ('relu2', nn.ReLU()),
                ('flatten', nn.Flatten()),
                ('lin1', Lin(245, 128)),#245
                ('relu3', nn.ReLU()),
                ]))

        self.conv3 = APPNP(K=2, alpha=0.8)
        #self.conv3 = SGConv(128, 128, K=2)
        self.norm = BatchNorm(128)
        self.mlp = nn.Sequential(OrderedDict([
                ('lin2', Lin(128, 64)),
                ('relu3', nn.ReLU()),
                ('lin3',Lin(64, 5))
                ]))

        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        graph_x = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        node_x = self.node_encode(x) # x: (3, 20, 20)
        node_x = self.conv3(node_x, edge_index, edge_weight)
        node_x = self.norm(node_x)
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.mlp(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        self.reserve = []
        with torch.no_grad():
            for param in self.parameters():
                self.reserve.append(copy.deepcopy(param))
                param.uniform_(-1.0, 1.0)

    def restore_parameters(self):
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                param.set_(self.reserve[i])


if __name__ == '__main__':

    args = parse_args()
    test_dataset = Visual_Genome(args.data_path, mode='testing')
    val_dataset = Visual_Genome(args.data_path, mode='evaluation')
    train_dataset = Visual_Genome(args.data_path, mode='training')
    if args.random_label:
        for dataset in [test_dataset, val_dataset, train_dataset]:
            for g in dataset:
                g.y.fill_(random.choice([i for i in range(5)]))
    model = VGNet().to(device)
    print(test_dataset[0].x.size())
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             pin_memory=True,
                             shuffle=False
                             )
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            pin_memory=True,
                            shuffle=False
                            )
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              pin_memory=True,
                              shuffle=True
                              )

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=15,
                                  min_lr=1e-5
                                  )

    min_error = None

    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        loss = Gtrain(train_loader,
                      model,
                      optimizer,
                      criterion=nn.CrossEntropyLoss()
                      )

        _, train_acc = Gtest(train_loader,
                             model,
                             criterion=nn.CrossEntropyLoss()
                             )

        val_error, val_acc = Gtest(val_loader,
                                   model,
                                   criterion=nn.CrossEntropyLoss()
                                   )
        test_error, test_acc = Gtest(test_loader,
                                     model,
                                     criterion=nn.CrossEntropyLoss()
                                     )
        scheduler.step(val_error)
        if min_error is None or val_error <= min_error:
            min_error = val_error

        t2 = time.time()

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader,
                                         model,
                                         criterion=nn.CrossEntropyLoss()
                                         )
            t3 = time.time()
            print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, '
                  'Test acc: {:.5f}'.format(epoch, t3 - t1, lr, loss, test_error, test_acc))
            continue

        print('Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Train acc: {:.5f}, Validation Loss: {:.5f}, '
              'Validation acc: {:5f}'.format(epoch, t2 - t1, lr, loss, train_acc, val_error, val_acc))

        torch.cuda.empty_cache()
    if args.random_label:
        save_path = 'vg_net_rd.pt'
    else:
        save_path = 'vg_net.pt'
    torch.save(model, args.model_path + save_path)

# Epoch  99[7.940s]: LR: 0.00020, Loss: 0.02630, Train acc: 0.99726, Validation Loss: 1.59648, Validation acc: 0.640000
# Epoch 100[8.275s]: LR: 0.00020, Loss: 0.02531, Test Loss: 1.52903, Test acc: 0.64000
# label random
# Epoch  99[8.487s]: LR: 0.00013, Loss: 0.03879, Train acc: 0.99835, Validation Loss: 4.21342, Validation acc: 0.210000
# Epoch 100[8.798s]: LR: 0.00013, Loss: 0.03675, Test Loss: 4.14552, Test acc: 0.18000