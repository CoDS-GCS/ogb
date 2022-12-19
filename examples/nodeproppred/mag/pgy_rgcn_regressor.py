import argparse
import os.path as osp
import pandas as pd
import torch
from resource import *
import datetime
import torch.nn.functional as F
import shutil
# from torch_geometric.datasets import Entities
from examples.nodeproppred.mag.EntitiesMetaSampler import EntitiesMetaSampler as Entities
from torch_geometric.nn import RGCNConv
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import  GraphSAINTRandomWalkSampler, GraphSAINTTaskBaisedRandomWalkSampler,GraphSAINTTaskWeightedRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
import sys
sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator, PygNodePropPredDataset_hsh
from torch_geometric.nn import MessagePassing
import sys
import os
import psutil
from pathlib import Path
from examples.nodeproppred.mag.GenerateSubgraphNodesScores import generateSubgraphNodeScoresFromdDF,generateTargetLabedlSubgraph,generateSubgraphNodeScores_Oversmoothing_FromdDF,generateSubgraphNodeScores_PPR_FromdDF

import pandas as pd
import operator
subject_node='Paper'
class Net(torch.nn.Module):
    def __init__(self,num_nodes=400000,num_relations=70):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(num_nodes, 32, num_relations, num_bases=30)
        # self.conv2 = RGCNConv(16, dataset.num_classes, dataset.num_relations, num_bases=30)
        self.conv2 = RGCNConv(32, 8, num_relations, num_bases=30)
        self.conv3 = RGCNConv(8, 1, num_relations, num_bases=30)

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.conv3(x, edge_index, edge_type)
        # return F.log_softmax(x, dim=1)
        # print(x)
        return x

def getDecodedSubgraph(org_dataset, subgraph):
    triples_list = []
    node_types = list(org_dataset.num_nodes_dict.keys())
    edge_types = list(org_dataset.edge_index_dict.keys())
    for idx in range(0, subgraph.edge_index.shape[1]):
        triples_list.append([
            edge_types[subgraph.edge_attr[idx]][0],  ## Src node type
            subgraph.local_node_idx[subgraph.edge_index[0][idx]].item(),  # Src node ID
            edge_types[subgraph.edge_attr[idx]][1],  # relation type
            edge_types[subgraph.edge_attr[idx]][2],  # Dest Node Type
            subgraph.local_node_idx[subgraph.edge_index[1][idx]].item()  # Dest Node ID
        ])
    return triples_list
    ##return global edge/node ids
def getSubgraphNodes(org_dataset, subgraph):
    triples_list = []
    node_types = list(org_dataset.num_nodes_dict.keys())
    edge_types = list(org_dataset.edge_index_dict.keys())
    for idx in range(0, subgraph.edge_index.shape[1]):
        triples_list.append([
            edge_types[subgraph.edge_attr[idx]][0],  ## Src node type
            subgraph.edge_index[0][idx].item(),  # Src node ID
            edge_types[subgraph.edge_attr[idx]][1],  # relation type
            subgraph.edge_attr[idx].item(),
            edge_types[subgraph.edge_attr[idx]][2],  # Dest Node Type
            subgraph.edge_index[1][idx].item()  # Dest Node ID
        ])
    return triples_list

def test(model,data):
    model.eval()
    out = model(data.edge_index, data.edge_type)
    # pred = out[data.test_idx].max(1)[1]
    # acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    acc=mean_squared_error(data.test_y,torch.flatten(out[data.test_idx]).detach())
    return acc,torch.flatten(out[data.test_idx]).detach(),data.test_y

model,data=None,None
def train(loaders_datasets,nepochs=100):
    datasets=[]
    sample_idx=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for loaderds in loaders_datasets:
        sample_idx =0
        print('loaderds=',loaderds['name'])
        for data in loaderds['train_loader']:
            print("data=", data)
            DecodedSubgraph_lst = getSubgraphNodes(loaderds['org_dataset'], data)
            DecodedSubgraph_df = pd.DataFrame(DecodedSubgraph_lst,columns=['Src_Node_Type', 'Src_Node_ID', 'Rel_type', 'Rel_ID',
                                                       'Dest_Node_Type', 'Dest_Node_ID'])
            DecodedSubgraph_df.to_csv("FM_3small_Subgraphs/"+loaderds['name']+"_sample_"+str(sample_idx)+"_biased.csv")
            # DecodedSubgraph_df.to_csv("FM_3small_Subgraphs/" + loaderds['name'] + "_sample_" + str(sample_idx) + "_UniformRW.csv")
            # normalized_df,all_labels_df,train_df,test_df,nt_df=generateSubgraphNodeScores_Oversmoothing_FromdDF(DecodedSubgraph_df,'rec')
            # all_labels_df, train_df, test_df, nt_df = generateSubgraphNodeScores_PPR_FromdDF(DecodedSubgraph_df, loaderds['subject_node'])
            # # all_labels_df, train, test, nt_df = generateTargetLabedlSubgraph(DecodedSubgraph_df, 'rec')
            # dataset = Entities(MaxNodeCount=40000, Triples_df=nt_df, labels_df=all_labels_df, Train_df=train_df, Test_df=test_df)
            # sampled_data=dataset.data.to(device)
            # datasets.append(sampled_data)
            sample_idx +=1
            print("sample_idx=",sample_idx)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    out, true=None,None
    for epoch in range(0, nepochs):
        ds_idx=0
        for data in datasets:
            model.train()
            optimizer.zero_grad()
            out = model(data.edge_index, data.edge_type)
            # F.nll_loss(out[data.train_idx], data.train_y).backward()
            F.mse_loss(torch.flatten(out[data.train_idx]), data.train_y.float()).backward()
            optimizer.step()
            test_acc, out, true = test(model,data)
            print('Dataset {:02d}, Epoch: {:02d}, MSE: {:.4f}'.format(ds_idx,epoch, test_acc))
            ds_idx+=1

    # test_acc, out, true = test(model,data)
    # print("out", out)
    # print("true", true)
    # print("lt elements=", torch.lt(true, out).sum() / len(true))
    return model

def saveModel(model,path,name):
    torch.save(model.state_dict(), path +  name + ".pt")
def loadModel(model,path,name="DBLP_FG_GS_SubgraphNodes_1"):
    if model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net().to(device)
    model.load_state_dict(torch.load(path +  name + ".pt"))
    model.eval()
    return model
def getTopKNodes(model,data,K):
    out = torch.flatten(model(data.edge_index, data.edge_type)).tolist()
    true_index = data.train_idx.tolist()
    true_index.extend(data.test_idx.tolist())
    dic = dict(map(lambda i, j: (i, j), true_index, out))
    sorted_d = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))
    keys_lst=list(sorted_d.keys())
    # print('Dictionary in descending order by value : ', sorted_d)
    return keys_lst[:int(len(keys_lst)*(K/100))],sorted_d


def exportModelOutput(model,data,path,name):
    true=out=None,None
    test_acc, out, true = test(model,data)
    print("out", out)
    print("true", true)
    print("lt elements=", torch.lt(true, out).sum() / len(true))
    out = torch.flatten(model(data.edge_index, data.edge_type))
    ###################################
    true_index=data.train_idx.tolist()
    true_index.extend(data.test_idx.tolist())
    true = data.train_y.tolist()
    true.extend(data.test_y.tolist())
    true_dic = {true_index[i]: true[i] for i in range(len(true_index))}
    temp_dict = sorted(list(true_dic.keys()))
    true_dic = {key: true_dic[key] for key in temp_dict}
    true= list(true_dic.values())
    pd.DataFrame({"node_idx":list(true_dic.keys()),"pred":out.tolist()[:-1],"true":true}).to_csv( path+name+"_modelout.csv",index=None)
def getDatasets(paths,ds_names,splits):
        datasets=[]
        to_remove_pedicates = []
        to_remove_subject_object = []
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
            train_loader = GraphSAINTTaskBaisedRandomWalkSampler(homo_data,
                                                                 # train_loader = GraphSAINTRandomWalkSampler(homo_data,
                                                                 batch_size=args.batch_size,
                                                                 walk_length=args.num_layers,
                                                                 Subject_indices=local2global[subject_node],
                                                                 num_steps=args.num_steps,
                                                                 sample_coverage=0,
                                                                 save_dir=dataset.processed_dir)

            # train_loader = GraphSAINTRandomWalkSampler(homo_data,
            #                                            batch_size=args.batch_size,
            #                                            walk_length=args.num_layers,
            #                                            num_steps=args.num_steps,
            #                                            sample_coverage=0,
            #                                            save_dir=dataset.processed_dir)
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
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=10)
# parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--runs', type=int, default=3)
# parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--walk_length', type=int, default=10)
parser.add_argument('--num_steps', type=int, default=20)
parser.add_argument('--loadTrainedModel', type=int, default=0)
parser.add_argument('--graphsaint_dic_path', type=str, default='none')
args = parser.parse_args()

def main():
    datasets = getDatasets([
        # '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_DBLP/',
        # '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_YAGO/',
        # '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_MAG/',
        '/media/hussein/UbuntuData/OGBN_Datasets/KGTOSA_MAG/'
    ],
        [
            # 'DBLP_Paper_Venue_FM_Literals2Nodes_SY1900_EY2021_50Class',
            # 'YAGO_FM51',
            # 'mag',
            'KGTOSA_MAG_StarQuery'
            # 'KGTOSA_MAG_Paper_Venue_2HOPS'
        ],
        [
            # 'time',
            # 'random',
            # 'time',
            'time',
            # 'time',
        ]
    )
    start_t = datetime.datetime.now()
    print(getrusage(RUSAGE_SELF).ru_maxrss)
    model=train(datasets,20)
    path="/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/"
    # saveModel(model,path,"DBLP_FG_GS_SubgraphNodes_Ls_Oversmoothing")
    saveModel(model, path, "DBLP_FG_GS_SubgraphNodes_Ls_PPR")
    print(getrusage(RUSAGE_SELF).ru_maxrss)
    print("train_time=",datetime.datetime.now()-start_t)
    #####################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model=loadModel(model,path,"DBLP_FG_GS_SubgraphNodes_MetaSampler")
    total_acc=0
    name=""
    for i in range(1,20):
        dataset = Entities(path+name+str(i), name+str(i), MaxNodeCount=40000)
        data = dataset[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data=data.to(device)
        keys,dict=getTopKNodes(model,data,75)
        test_acc, out, true = test(model,data)
        total_acc+=test_acc
        print("test_acc", test_acc)
    print("avg=",total_acc/13)
if __name__ == "__main__":
    main()