from copy import copy
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
import datetime
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import  GraphSAINTRandomWalkSampler , GraphSAINTTaskBaisedRandomWalkSampler, ShaDowKHopSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
import sys
import os
import psutil
from pathlib import Path
import pandas as pd
import random
import statistics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import traceback
import sys
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator, PygNodePropPredDataset_hsh
from resource import *
from logger import Logger
import faulthandler
from sklearn.metrics import mean_squared_error
from examples.nodeproppred.mag.GenerateSubgraphNodesScores import generateSubgraphNodeScoresFromdDF,generateTargetLabedlSubgraph,generateSubgraphNodeScoresFromTriplesdf,generateSubgraphNodeScores_Oversmoothing_FromTriplesdf
from examples.nodeproppred.mag.EntitiesMetaSampler import EntitiesMetaSampler as Entities
from examples.nodeproppred.mag.pgy_rgcn_regressor import loadModel,getTopKNodes
faulthandler.enable()
import pickle

subject_node = 'Paper'


def print_memory_usage():
    # print("max_mem_GB=",psutil.Process().memory_info().rss / (1024 * 1024*1024))
    # print("get_process_memory=",getrusage(RUSAGE_SELF).ru_maxrss/(1024*1024))
    print('used virtual memory GB:', psutil.virtual_memory().used / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.rel_lins = ModuleList([
            Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])

        self.root_lins = ModuleList([
            Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type, node_type):
        out = x.new_zeros(x.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])

        return out

    def message(self, x_j, edge_type: int):
        return self.rel_lins[edge_type](x_j)


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H = in_channels, hidden_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, 1, self.num_node_types, num_edge_types))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx):
        # Create global node feature matrix.
        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device
        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
        for i, conv in enumerate(self.convs):
            out_dict = {}
            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)
            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                ################## fill missed rows hsh############################
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)
            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])
            x_dict = out_dict
        return x_dict
dic_results = {}

def graphSaint():
    def getSubgraphNodes(org_dataset, subgraph,ignore_rel_inverse=True):
        triples_list = []
        node_types = list(org_dataset.num_nodes_dict.keys())
        edge_types = list(org_dataset.edge_index_dict.keys())
        if ignore_rel_inverse == False:
            for idx in range(0, subgraph.edge_index.shape[1]):
                    triples_list.append([
                        edge_types[subgraph.edge_attr[idx]][0],  ## Src node type
                        subgraph.edge_index[0][idx].item(),  # Src node ID
                        edge_types[subgraph.edge_attr[idx]][1],  # relation type
                        subgraph.edge_attr[idx].item(),
                        edge_types[subgraph.edge_attr[idx]][2],  # Dest Node Type
                        subgraph.edge_index[1][idx].item()  # Dest Node ID
                    ])
        else:
            for idx in range(0, subgraph.edge_index.shape[1]):
                if  edge_types[subgraph.edge_attr[idx]][1].startswith("inv_")==False:
                    triples_list.append([
                        edge_types[subgraph.edge_attr[idx]][0],  ## Src node type
                        subgraph.edge_index[0][idx].item(),  # Src node ID
                        edge_types[subgraph.edge_attr[idx]][1],  # relation type
                        subgraph.edge_attr[idx].item(),
                        edge_types[subgraph.edge_attr[idx]][2],  # Dest Node Type
                        subgraph.edge_index[1][idx].item()  # Dest Node ID
                    ])
        return triples_list
    def labelSubgraphWithScores(data,org_dataset,subject_node,local2global):
        ####################Local to global node id #########################
        node_types = list(org_dataset.num_nodes_dict.keys())
        edge_types = list(org_dataset.edge_index_dict.keys())
        triples_types=[list(org_dataset.edge_index_dict.keys())[x] for x in data.edge_attr.tolist()]
        pad_val=100000000
        s_type_lst=[node_types.index(x[0])*pad_val  for x in triples_types]
        s_global_lst=[x + y for x, y in zip(s_type_lst, data.edge_index[0].tolist())]
        # p_type_lst = [edge_types.index(x) for x in triples_types]
        o_type_lst = [node_types.index(x[2])*pad_val for x in triples_types]
        o_global_lst = [x + y for x, y in zip(o_type_lst, data.edge_index[1].tolist())]
        all_nodes_ids = [x*pad_val+y for x, y in zip(data.node_type.tolist(), data.local_node_idx.tolist())]
        triples_df = pd.DataFrame(list(zip(s_global_lst,data.edge_attr.tolist(),o_global_lst)))
        target_node_idx = list(org_dataset.num_nodes_dict.keys()).index(subject_node)
        target_nodes_idx_lst = ((data.node_type == target_node_idx).nonzero(as_tuple=True)[0]).tolist()
        target_nodes_idx_lst=[node_types.index(subject_node)*pad_val+ x for x in target_nodes_idx_lst]
        ####################Calc Node Score #########################
        normalized_df, all_labels_df, train, test = generateSubgraphNodeScores_Oversmoothing_FromTriplesdf(triples_df,all_nodes_ids,target_nodes_idx_lst)
        ################### append is_target node score ##############
        scores_lst = all_labels_df["NScore"].tolist()
        ### set target Node Score to max+100 ############
        scores_lst.append(max(scores_lst)+100)
        data.y = torch.reshape(torch.tensor(scores_lst), (len(scores_lst), 1))
        ######################## build New Train Mask###################
        data.train_mask = torch.tensor([True if x in train else False for x in range(0, len(data.y))])
        ####### increase Nodes num by 1 ( the is target node ) ##########
        data.num_nodes += 1
        ############## add the is target node type ################
        data.node_type = torch.cat([data.node_type, torch.tensor([len(org_dataset.num_nodes_dict.keys())])], dim=-1)
        ############## add the is target node local index to 0 ################
        data.local_node_idx = torch.cat([data.local_node_idx, torch.tensor([0])], dim=-1)
        ############## build new triples by labling the target nodes with a new edges ################
        o_target = torch.tensor([max((data.edge_index[0].max(), data.edge_index[1].max())).item() + 1] * len(target_nodes_idx_lst))
        s_target = torch.tensor(target_nodes_idx_lst)
        p_target = torch.tensor([len(org_dataset.edge_index_dict.keys())] * len(target_nodes_idx_lst))
        data.edge_index = torch.tensor([torch.cat([data.edge_index[0], s_target], dim=0).tolist(),torch.cat([data.edge_index[1], o_target], dim=0).tolist()])
        data.edge_attr = torch.cat([data.edge_attr, p_target], dim=0)
        return data
    def train(epoch,datasets):
        model.train()
        # tqdm.monitor_interval = 0
        # pbar = tqdm(total=args.num_steps * args.batch_size)
        # print("len(train_loader)",len(train_loader))
        for ds in datasets:
            print("ds name",ds['name'])
            pbar = tqdm(total=len(ds['train_loader']))
            pbar.set_description(f'Epoch {epoch:02d}')
            total_loss= total_examples = 0
            for data in ds['train_loader']:
                data = labelSubgraphWithScores(data, ds['org_dataset'], ds['subject_node'],ds['local2global'])
                print("data=",data)
                data = data.to(device)
                for i in range(0,10):
                    optimizer.zero_grad()
                    out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,data.local_node_idx)
                    out = out[data.train_mask].squeeze(1)
                    # out = torch.index_select(out, 0, data.root_n_id)
                    y = data.y.squeeze(1)[data.train_mask]
                    loss = F.mse_loss(out, y.float())
                    print("loss=",loss)
                    loss.backward()
                    optimizer.step()
                    total_loss= loss.item()
                pbar.update(1)
            pbar.refresh()  # force print final state
            pbar.close()
        return total_loss

    @torch.no_grad()
    def test(datasets):
        model.eval()
        for dataset in datasets:
            out = model.inference(dataset['x_dict'], dataset['edge_index_dict'], dataset['key2int'])
            out = out[key2int[dataset['subject_node']]]
            y_true =dataset['data'].y_dict[dataset['subject_node']]
            train_mse = mean_squared_error(y_true[dataset['split_idx']['train'][subject_node]].float(),
                                           out[dataset['split_idx']['train'][subject_node]].detach())
            test_mse = mean_squared_error(y_true[dataset['split_idx']['test'][subject_node]].float(),
                                          out[dataset['split_idx']['test'][subject_node]].detach())
            valid_mse = mean_squared_error(y_true[dataset['split_idx']['valid'][subject_node]].float(),
                                           out[dataset['split_idx']['valid'][subject_node]].detach())
        return train_mse, test_mse, valid_mse
    def getDatasets(paths,ds_names,splits):
        datasets=[]
        for i in range(len(paths)):
            dataset_dic={}
            dataset = PygNodePropPredDataset_hsh(name=ds_names[i],# root='/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_MAG/',
                                                 root=paths[i],numofClasses=str(50))
            start_t = datetime.datetime.now()
            org_dataset = data = dataset[0]
            dataset_dic["name"] = ds_names[i]
            dataset_dic["org_dataset"] = org_dataset
            global subject_node
            subject_node = list(data.y_dict.keys())[0]
            dataset_dic["subject_node"]=subject_node
            split_idx = dataset.get_idx_split(splits[i])
            dataset_dic["split_idx"] = split_idx
            end_t = datetime.datetime.now()
            # We do not consider those attributes for now.
            data.node_year_dict = None
            data.edge_reltype_dict = None
            to_remove_rels = []
            for keys, (row, col) in data.edge_index_dict.items():
                if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
                    to_remove_rels.append(keys)
            for keys, (row, col) in data.edge_index_dict.items():
                if (keys[1] in to_remove_pedicates):
                    to_remove_rels.append(keys)
                    to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))

            for elem in to_remove_rels:
                data.edge_index_dict.pop(elem, None)
                data.edge_reltype.pop(elem, None)

            for key in to_remove_subject_object:
                data.num_nodes_dict.pop(key, None)
            ##############add inverse edges ###################
            edge_index_dict=data.edge_index_dict
            dataset_dic["edge_index_dict"]=edge_index_dict
            key_lst = list(edge_index_dict.keys())
            for key in key_lst:
                r, c = edge_index_dict[(key[0], key[1], key[2])]
                edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

            print("data=", data)
            dataset_dic["data"] = data
            out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
            edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
            dataset_dic["key2int"] = key2int
            dataset_dic["edge_index"] = edge_index
            dataset_dic["local2global"]=local2global
            # print_memory_usage()

            homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                             node_type=node_type, local_node_idx=local_node_idx,
                             num_nodes=node_type.size(0))

            homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
            homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]
            homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
            homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
            dataset_dic["homo_data"]=homo_data
            print(homo_data)
            start_t = datetime.datetime.now()
            print("dataset.processed_dir", dataset.processed_dir)
            train_loader = GraphSAINTRandomWalkSampler(homo_data,
                                                       batch_size=args.batch_size,
                                                       walk_length=args.num_layers,
                                                       num_steps=args.num_steps,
                                                       sample_coverage=0,
                                                       save_dir=dataset.processed_dir)
            dataset_dic["train_loader"]=train_loader

            feat = torch.Tensor(data.num_nodes_dict[subject_node], 128)
            torch.nn.init.xavier_uniform_(feat)
            feat_dic = {subject_node: feat}
            ################################################################
            x_dict = {}
            for key, x in feat_dic.items():
                x_dict[key2int[key]] = x
            dataset_dic["x_dict"] = x_dict
            end_t = datetime.datetime.now()
            print("Sampling time=", end_t - start_t, " sec.")
            datasets.append(dataset_dic)
        return datasets
    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    parser.add_argument('--graphsaint_dic_path', type=str, default='none')
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    args = parser.parse_args()
    GSAINT_Dic = {}
    to_remove_pedicates = []
    to_remove_subject_object = []
    to_keep_edge_idx_map = []
    GA_Index = 0
    MAG_datasets = ["mag"]
    print(args)
    gsaint_Final_Test = 0
    logger = Logger(args.runs, args)
    for GA_dataset_name in MAG_datasets:
        try:
            gsaint_start_t = datetime.datetime.now()
            datasets=getDatasets([
                                  '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_DBLP/',
                                  '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_YAGO/',
                                   '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_MAG/'
                                ],
                                 [
                                  'DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class',
                                  'YAGO_FM51',
                                   'mag',
                                 ],
                                 [
                                     'time',
                                     'random',
                                     'time',
                                 ])
            start_t = datetime.datetime.now()
            # Map informations to their canonical type.
            #######################intialize random features ###############################
            data=datasets[0]['data']
            key2int=datasets[0]['key2int']
            edge_index_dict=datasets[0]['edge_index_dict']
            edge_index_dict = datasets[0]['edge_index_dict']
            x_dict=datasets[0]['x_dict']
            num_nodes_dict = {}
            for key, N in data.num_nodes_dict.items():
                num_nodes_dict[key2int[key]] = N

            end_t = datetime.datetime.now()
            print("model init time CPU=", end_t - start_t, " sec.")
            device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
            model = RGCN(128, args.hidden_channels,args.num_layers,
                         args.dropout, num_nodes_dict, list(x_dict.keys()),
                         len(edge_index_dict.keys())).to(device)

            x_dict = {k: v.to(device) for k, v in x_dict.items()}
            y_true = data.y_dict[datasets[0]['subject_node']]
            print("x_dict=", x_dict.keys())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            if args.loadTrainedModel == 1:
                model.load_state_dict(torch.load("ogbn-DBLP-FM-GSaint.model"))
                model.eval()
                out = model.inference(x_dict, edge_index_dict, key2int)
                out = out[key2int[subject_node]]
                y_pred = out.argmax(dim=-1, keepdim=True).cpu()
                y_true = data.y_dict[subject_node]

                out_lst = torch.flatten(y_true).tolist()
                pred_lst = torch.flatten(y_pred).tolist()
                out_df = pd.DataFrame({"y_pred": pred_lst, "y_true": out_lst})
                out_df.to_csv("GSaint_DBLP_conf_output.csv", index=None)
            else:
                print("start test")
                # test(dataset[0])  # Test if inference on GPU succeeds.
                total_run_t = 0
                for run in range(args.runs):
                    start_t = datetime.datetime.now()
                    model.reset_parameters()
                    for epoch in range(1, 1 + args.epochs):
                        loss = train(epoch,datasets)
                        ##############
                        if loss == -1:
                            return 0.001
                            ##############
                        torch.cuda.empty_cache()
                        result = test(datasets)
                        logger.add_result(run, result)
                        train_acc, valid_acc, test_acc = result
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}, '
                              f'Valid: {100 * valid_acc:.2f}, '
                              f'Test: {100 * test_acc:.2f}')
                    logger.print_statistics(run)
                    end_t = datetime.datetime.now()
                    total_run_t = total_run_t + (end_t - start_t).total_seconds()
                    print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
                    print(getrusage(RUSAGE_SELF))
                Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
                gsaint_Final_Test = Final_Test.item()
                # pd.DataFrame(dic_results).transpose().to_csv("/shared_mnt/KGTOSA_MAG/GSAINT_" + GA_dataset_name + "_Times.csv", index=False)
                # shutil.rmtree("/shared_mnt/DBLP/" + dataset_name)
                torch.save(model.state_dict(), "/media/hussein/UbuntuData/OGBN_Datasets/GNN-MetaSampler_GS.model")
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("dataset_name Exception")
    return gsaint_Final_Test


if __name__ == '__main__':
    print(graphSaint())






