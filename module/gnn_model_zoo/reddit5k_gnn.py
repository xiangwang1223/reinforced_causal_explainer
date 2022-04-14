import time
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import Linear as Lin, ReLU, Softmax,Tanh

from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, BatchNorm, GraphConv
from torch.optim.lr_scheduler import ReduceLROnPlateau

from module.utils import set_seed, Gtrain, Gtest
from module.data_loader_zoo.reddit5k_dataloader import Reddit5k

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Reddit-Multi-5k Model")

    parser.add_argument('--data_path', nargs='?', default='../../Data/Reddit5k',
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default='../../params/',
                        help='path for saving trained model.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--random_label', type=bool, default=True,
                        help='train a model under label randomization for sanity check')

    return parser.parse_args()


class Reddit5kNet(torch.nn.Module):
    def __init__(self):
        super(Reddit5kNet, self).__init__()

        # node encoder
        self.lin0 = Lin(1, 32)
        self.tanh0 = Tanh()
        self.conv_block = nn.ModuleList(
            [GraphConv(32, 32) for _ in range(3)]
        )

        self.relus = nn.ModuleList(
            [ReLU() for _ in range(3)]
        )
        self.batch_norms = nn.ModuleList(
            [BatchNorm(32) for _ in range(2)]
        )
        self.lin1 = torch.nn.Linear(32, 64)
        self.relu0 = ReLU()
        self.lin2 = torch.nn.Linear(64, 5)

    def forward(self, x, edge_index, edge_attr, batch):

        graph_x = self.get_graph_rep(x, edge_index, edge_attr, batch)
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_weight, batch):

        x = self.tanh0(self.lin0(x))
        for ReLU, conv, norm in zip(
                self.relus, self.conv_block, self.batch_norms):
            x = ReLU(conv(x, edge_index, edge_weight))
            x = norm(x)

        return x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):

        edge_weight = edge_attr.view(-1)
        node_x = self.get_node_reps(x, edge_index, edge_weight, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu0(self.lin1(graph_x))
        pred = self.lin2(pred)
        softmax = Softmax(dim=1)
        self.readout = softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

if __name__ == '__main__':

    args = parse_args()
    set_seed(0)
    train_dataset = Reddit5k(args.data_path, mode='training')
    val_dataset = Reddit5k(args.data_path, mode='evaluation')
    test_dataset = Reddit5k(args.data_path, mode='testing')

    if args.random_label:
        for dataset in [test_dataset, val_dataset, train_dataset]:
            for g in dataset:
                g.y.fill_(random.choice([i for i in range(5)]))

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
    model = Reddit5kNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr
                                 )
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.8,
                                  patience=20,
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

    if args.random_label:
        save_path = 'reddit5k_net_rd.pt'
    else:
        save_path = 'reddit5k_net.pt'
    torch.save(model, args.model_path + save_path)

# Epoch 199[4.027s]: LR: 0.00051, Loss: 0.05508, Train acc: 0.98350, Validation Loss: 0.03881, Validation acc: 0.982000
# Epoch 200[4.247s]: LR: 0.00051, Loss: 0.05303, Test Loss: 0.05021, Test acc: 0.97800
# data random
# Epoch 199[4.247s]: LR: 0.00017, Loss: 1.58435, Train acc: 0.25831, Validation Loss: 1.61846, Validation acc: 0.232000
# Epoch 200[4.526s]: LR: 0.00017, Loss: 1.58347, Test Loss: 1.63560, Test acc: 0.15800