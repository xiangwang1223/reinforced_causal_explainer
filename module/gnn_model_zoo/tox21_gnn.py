
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import CGConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Linear as Lin, ReLU, Softmax

from module.utils import set_seed, Gtrain, Gtest
from module.data_loader_zoo.tox21_dataloader import Tox21

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tox21_AhR Model")

    parser.add_argument('--data_path', nargs='?', default='../../Data/Tox21',
                        help='Input data path.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--num_unit', type=int, default=2,
                        help='number of Convolution layers(units)')

    return parser.parse_args()



class Tox21Net(torch.nn.Module):

    def __init__(self, num_unit=2, dim=75):
        super(Tox21Net, self).__init__()

        self.num_unit = num_unit
        self.lin0 = torch.nn.Linear(53, dim)

        self.relus = nn.ModuleList(
            [ReLU() for _ in range(num_unit + 3)]
        )

        self.dropout = nn.Dropout(p=0.2)
        self.conv = CGConv(channels=dim, dim=4)
        self.lin1 = Lin(dim, 128)
        #self.set2set = Set2Set(dim, processing_steps=3) # global pooling layer
        self.lin2 = Lin(128, 32)
        self.lin3 = Lin(32, 2)

        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):

        node_x = self.get_node_reps(x, edge_index, edge_attr)
        graph_x = self.relus[-1](global_max_pool(node_x, batch))
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr):

        x = self.dropout(x)
        x = self.relus[0](self.lin0(x))
        for i in range(1, self.num_unit + 1):
            x = self.relus[i](self.conv(x, edge_index, edge_attr))

        return self.lin1(x)

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr)
        graph_x = self.relus[-2](global_max_pool(node_x, batch)) # self.set2set(node_x, batch))
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relus[-1](self.lin2(graph_x))
        pred = self.lin3(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == '__main__':

    args = parse_args()
    set_seed(0)
    train_dataset = Tox21(args.data_path, mode='training')
    val_dataset = Tox21(args.data_path, mode='evaluation')
    test_dataset = Tox21(args.data_path, mode='testing')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False
                             )
    model = Tox21Net(args.num_unit).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=5,
                                  min_lr=1e-4
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
    torch.save(model, '../../params/tox21_net.pt')
# Epoch 199[4.841s]: LR: 0.00010, Loss: 0.20510, Train acc: 0.92068, Validation Loss: 0.25100, Validation acc: 0.892916
# Epoch 200[4.969s]: LR: 0.00010, Loss: 0.20241, Test Loss: 0.32355, Test acc: 0.85662