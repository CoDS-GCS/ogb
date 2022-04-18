import argparse

import pandas  as pd
import torch
import torch.nn.functional as F
import datetime
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
import shutil
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator,PygNodePropPredDataset_hsh

from logger import Logger
from resource import *

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    print(getrusage(RUSAGE_SELF))
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    parser = argparse.ArgumentParser(description='OGBN-MAG (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    # parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_sage', type=int,default = False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # fieldOfStudy_Coverage_df = pd.read_csv("/media/hussein/UbuntuData/OGBN_Datasets/ogbn_mag_fieldOfStudy_Coverage_top_10000.csv")
    fieldOfStudy_Coverage_df = pd.read_csv("/media/hussein/UbuntuData/OGBN_Datasets/ogbn_mag_FM_Usecases.csv")
    fieldOfStudy_Coverage_df = fieldOfStudy_Coverage_df[fieldOfStudy_Coverage_df["do_train"] == 1].reset_index(
        drop=True)
    dic_results = {}
    for i, row in fieldOfStudy_Coverage_df.iterrows():
        gnn_start_t = datetime.datetime.now()
        dataset_name = ""
        if i >= 0:
            start_t = datetime.datetime.now()
            if str(row["topic"]) == "FM":
                dataset_name = "mag"
            else:
                dataset_name = "OBGN_MAG_Usecase_" + str(int(row["Q_idx"])) + "_" + str(str(row["topic"]).strip().replace(" ", "_").replace("/", "_"))
            print("dataset_name=", dataset_name)
            dic_results[dataset_name] = {}
            dic_results[dataset_name]["q_idx"] = int(row["Q_idx"])
            dic_results[dataset_name]["usecase"] = dataset_name
            dic_results[dataset_name]["gnn_hyper_params"] = str(args)

            start_t = datetime.datetime.now()
            dataset = PygNodePropPredDataset_hsh(name=dataset_name,root = '/media/hussein/UbuntuData/OGBN_Datasets/')
            # dataset = PygNodePropPredDataset(name='ogbn-mag-QM3')
            # dataset = PygNodePropPredDataset(name='ogbn-mag-QM2')
            # dataset = PygNodePropPredDataset(name='ogbn-mag-QM4')
            # dataset = PygNodePropPredDataset(name='ogbn-mag-QM1')
            rel_data = dataset[0]

            # We are only interested in paper <-> paper relations.
            feat = torch.Tensor(rel_data.num_nodes_dict['paper'], 128)
            torch.nn.init.xavier_uniform_(feat)
            # feat_dic = {'paper': feat}

            data = Data(
                # x=rel_data.x_dict['paper'],
                x=feat,
                edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
                y=rel_data.y_dict['paper'])
            dic_results[dataset_name]["data"] = str(data)
            data = T.ToSparseTensor()(data)
            data.adj_t = data.adj_t.to_symmetric()
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']['paper'].to(device)
            end_t = datetime.datetime.now()
            print("data load time=", end_t - start_t, " sec.")
            dic_results[dataset_name]["data_init_time"] = (end_t - start_t).total_seconds()
            start_t = datetime.datetime.now()
            model_name=""
            if args.use_sage:
                model_name="SAGE"
                model = SAGE(data.num_features, args.hidden_channels,
                             dataset.num_classes, args.num_layers,
                             args.dropout).to(device)
            else:
                model = GCN(data.num_features, args.hidden_channels,
                            dataset.num_classes, args.num_layers,
                            args.dropout).to(device)
                model_name="GCN"
                # Pre-compute GCN normalization.
                adj_t = data.adj_t.set_diag()
                deg = adj_t.sum(dim=1).to(torch.float)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
                data.adj_t = adj_t

            dic_results[dataset_name]["GNN_Model"] = model_name
            print("model_name=", model_name )
            data = data.to(device)
            evaluator = Evaluator(name='ogbn-mag')
            logger = Logger(args.runs, args)
            end_t = datetime.datetime.now()
            print("model init time CPU=", end_t - start_t, " sec.")
            dic_results[dataset_name]["model init Time"] = (end_t - start_t).total_seconds()
            model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            if args.loadTrainedModel==1:
                model.load_state_dict(torch.load("ogbn-mag-FM-GCN.model"))
                model.eval()
                out = model(data.x, data.adj_t)
                y_pred = out.argmax(dim=-1, keepdim=True)
                out_lst=torch.flatten(data.y).tolist()
                pred_lst = torch.flatten(y_pred).tolist()
                out_df = pd.DataFrame({"y_pred":pred_lst,"y_true":out_lst})
                # print(y_pred, data.y_dict['paper'])
                # print(out_df)
                out_df.to_csv("GCN_mag_output.csv",index=None)
            else:
                total_run_t = 0
                for run in range(args.runs):
                    start_t = datetime.datetime.now()
                    model.reset_parameters()
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    for epoch in range(1, 1 + args.epochs):
                        loss = train(model, data, train_idx, optimizer)
                        result = test(model, data, split_idx, evaluator)
                        logger.add_result(run, result)

                        if epoch % args.log_steps == 0:
                            train_acc, valid_acc, test_acc = result
                            print(f'Run: {run + 1:02d}, '
                                  f'Epoch: {epoch:02d}, '
                                  f'Loss: {loss:.4f}, '
                                  f'Train: {100 * train_acc:.2f}%, '
                                  f'Valid: {100 * valid_acc:.2f}% '
                                  f'Test: {100 * test_acc:.2f}%')

                    end_t = datetime.datetime.now()
                    logger.print_statistics(run)
                    total_run_t = total_run_t + (end_t - start_t).total_seconds()
                    print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
                    print(getrusage(RUSAGE_SELF))
                total_run_t = (total_run_t + 0.00001) / args.runs
                torch.save(model.state_dict(), "ogbn-mag-FM-GCN.model")
                gnn_end_t = datetime.datetime.now()
                Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
                model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
                dic_results[dataset_name]["init_ru_maxrss"] = init_ru_maxrss
                dic_results[dataset_name]["model_ru_maxrss"] = model_loaded_ru_maxrss
                dic_results[dataset_name]["model_trained_ru_maxrss"] = model_trained_ru_maxrss
                dic_results[dataset_name]["Highest_Train"] = Highest_Train.item()
                dic_results[dataset_name]["Highest_Valid"] = Highest_Valid.item()
                dic_results[dataset_name]["Final_Train"] = Final_Train.item()
                dic_results[dataset_name]["Final_Test"] = Final_Test.item()
                dic_results[dataset_name]["runs_count"] = args.runs
                dic_results[dataset_name]["avg_train_time"] = total_run_t
                dic_results[dataset_name]["gnn_total_time"] = (gnn_end_t - gnn_start_t).total_seconds()
                pd.DataFrame(dic_results).transpose().to_csv(
                    "/media/hussein/UbuntuData/OGBN_Datasets/OGBN_MAG_"+model_name+"_times" + ".csv", index=False)
                shutil.rmtree("/media/hussein/UbuntuData/OGBN_Datasets/" + dataset_name)
                torch.save(model.state_dict(),"/media/hussein/UbuntuData/OGBN_Datasets/" + dataset_name + "_"+model_name+"_QM.model")


if __name__ == "__main__":
    main()
