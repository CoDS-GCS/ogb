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
from torch_geometric.loader import  GraphSAINTRandomWalkSampler , GraphSAINTTaskBaisedRandomWalkSampler
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

sys.path.insert(0, '/shared_mnt/github_repos/ogb/')
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator, PygNodePropPredDataset_hsh
from resource import *
from logger import Logger


# import faulthandler;
# faulthandler.enable()
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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
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

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types))
        self.convs.append(RGCNConv(H, O, self.num_node_types, num_edge_types))

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
        #To Do
        # custom weight for each edge type
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, edge_index_dict, key2int):
        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)

        # paper_count=len(x_dict[2])
        # paper_count = len(x_dict[3])
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb
            # print(key," size=",x_dict[int(key)].size())

        # print(key2int)
        # print("x_dict keys=",x_dict.keys())

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)
            # print(key,adj_t_dict[key].size)

        for i, conv in enumerate(self.convs):
            out_dict = {}

            for j, x in x_dict.items():
                out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                # print("keys=",keys)
                # print("adj_t=",adj_t)
                # print("key2int[src_key]=",key2int[src_key])
                # print("x_dict[key2int[src_key]]=",x_dict[key2int[src_key]].size())
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                # print("out size=",out.size())
                # print("tmp size=",conv.rel_lins[key2int[keys]](tmp).size())
                ################## fill missed rows hsh############################
                tmp = conv.rel_lins[key2int[keys]](tmp).resize_([out.size()[0], out.size()[1]])
                out.add_(tmp)

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


dic_results = {}


def graphSaint(to_remove_pedicates, to_remove_subject_object, to_keep_edge_idx_map, GA_Index, dataset, GA_dataset_name,baised_sampler):
    def train(epoch):
        model.train()

        pbar = tqdm(total=args.num_steps * args.batch_size)
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_examples = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                        data.local_node_idx)
            out = out[data.train_mask]
            y = data.y[data.train_mask].squeeze()
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            pbar.update(args.batch_size)

        pbar.refresh()  # force print final state
        pbar.close()
        pbar.reset()
        return total_loss / total_examples

    @torch.no_grad()
    def test():
        model.eval()

        out = model.inference(x_dict, edge_index_dict, key2int)
        out = out[key2int['rec']]

        y_pred = out.argmax(dim=-1, keepdim=True).cpu()
        y_true = data.y_dict['rec']

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']['rec']],
            'y_pred': y_pred[split_idx['train']['rec']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']['rec']],
            'y_pred': y_pred[split_idx['valid']['rec']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']['rec']],
            'y_pred': y_pred[split_idx['test']['rec']],
        })['acc']
        return train_acc, valid_acc, test_acc

    parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.05)
    # parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20000)
    # parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--loadTrainedModel', type=int, default=0)
    init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
    args = parser.parse_args()
    print(args)
    gsaint_Final_Test = 0
    try:
        gsaint_start_t = datetime.datetime.now()
        dataset = init_graphSaintDataset('/home/hussein/Downloads/', GA_dataset_name, 50)
        dataset_name = GA_dataset_name + "_GA_" + str(GA_Index)
        print("dataset_name=", dataset_name)
        dic_results[dataset_name] = {}
        dic_results[dataset_name]["GNN_Model"] = "GSaint"
        dic_results[dataset_name]["GA_Index"] = GA_Index
        dic_results[dataset_name]["to_keep_edge_idx_map"] = to_keep_edge_idx_map
        dic_results[dataset_name]["usecase"] = dataset_name
        dic_results[dataset_name]["gnn_hyper_params"] = str(args)

        print(getrusage(RUSAGE_SELF))
        start_t = datetime.datetime.now()
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        end_t = datetime.datetime.now()
        print("dataset init time=", end_t - start_t, " sec.")
        dic_results[dataset_name]["GSaint_data_init_time"] = (end_t - start_t).total_seconds()
        evaluator = Evaluator(name='ogbn-mag')
        logger = Logger(args.runs, args)

        start_t = datetime.datetime.now()
        # We do not consider those attributes for now.
        data.node_year_dict = None
        data.edge_reltype_dict = None

        # remove_subject_object = [
        #     # 'doi',
        #     # 'sameAs',
        #     # 'rdf',
        #     # 'net',
        #     # 'differentFrom',
        #     # 'otherHomepage',
        #     # 'primaryElectronicEdition',
        #     # 'otherElectronicEdition',
        #     # 'archivedElectronicEdition',
        #     # 'entity',
        #     # 'db'
        # ]
        # remove_pedicates = [
        #     # 'schema#awardWebpage',
        #     # 'schema#webpage',
        #     # 'schema#archivedWebpage',
        #     # 'schema#primaryHomepage',
        #     # 'schema#wikipedia',
        #     # 'schema#orcid',
        #     # 'schema#publishedAsPartOf'
        # ]
        to_remove_rels = []
        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
                # print("to remove keys=",keys)
                to_remove_rels.append(keys)

        for keys, (row, col) in data.edge_index_dict.items():
            if (keys[1] in to_remove_pedicates):
                # print("to remove keys=",keys)
                to_remove_rels.append(keys)
                to_remove_rels.append((keys[2], 'inv_' + keys[1], keys[0]))

        for elem in to_remove_rels:
            data.edge_index_dict.pop(elem, None)
            data.edge_reltype.pop(elem, None)

        for key in to_remove_subject_object:
            data.num_nodes_dict.pop(key, None)

        dic_results[dataset_name]["data"] = str(data)
        ##############add inverse edges ###################
        edge_index_dict = data.edge_index_dict
        key_lst = list(edge_index_dict.keys())
        for key in key_lst:
            r, c = edge_index_dict[(key[0], key[1], key[2])]
            edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

        print("data after filter=", str(data))
        print_memory_usage()
        out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
        edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
        print_memory_usage()

        homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                         node_type=node_type, local_node_idx=local_node_idx,
                         num_nodes=node_type.size(0))

        homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
        homo_data.y[local2global['rec']] = data.y_dict['rec']

        homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
        homo_data.train_mask[local2global['rec'][split_idx['train']['rec']]] = True
        # print(homo_data)
        start_t = datetime.datetime.now()

        if baised_sampler:
            train_loader = GraphSAINTTaskBaisedRandomWalkSampler(homo_data,
                                                                 batch_size=args.batch_size,
                                                                 walk_length=args.num_layers,
                                                                 Subject_indices=local2global['rec'],
                                                                 num_steps=args.num_steps,
                                                                 sample_coverage=0,
                                                                 save_dir=dataset.processed_dir)
        else:
            train_loader = GraphSAINTRandomWalkSampler(homo_data,
                                                   batch_size=args.batch_size,
                                                   walk_length=args.num_layers,
                                                   num_steps=args.num_steps,
                                                   sample_coverage=0,
                                                   save_dir=dataset.processed_dir)


        end_t = datetime.datetime.now()
        print("Sampling time=", end_t - start_t, " sec.")
        dic_results[dataset_name]["GSaint_Sampling_time"] = (end_t - start_t).total_seconds()
        start_t = datetime.datetime.now()
        # Map informations to their canonical type.
        #######################intialize random features ###############################
        feat = torch.Tensor(data.num_nodes_dict['rec'], 128)
        torch.nn.init.xavier_uniform_(feat)
        feat_dic = {'rec': feat}
        ################################################################
        x_dict = {}
        # for key, x in data.x_dict.items():
        for key, x in feat_dic.items():
            x_dict[key2int[key]] = x

        num_nodes_dict = {}
        for key, N in data.num_nodes_dict.items():
            num_nodes_dict[key2int[key]] = N

        end_t = datetime.datetime.now()
        print("model init time CPU=", end_t - start_t, " sec.")
        dic_results[dataset_name]["model init Time"] = (end_t - start_t).total_seconds()
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
                     args.dropout, num_nodes_dict, list(x_dict.keys()),
                     len(edge_index_dict.keys())).to(device)

        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        print("x_dict=", x_dict.keys())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model_loaded_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
        if args.loadTrainedModel == 1:
            model.load_state_dict(torch.load("ogbn-DBLP-FM-GSaint.model"))
            model.eval()
            out = model.inference(x_dict, edge_index_dict, key2int)
            out = out[key2int['rec']]
            y_pred = out.argmax(dim=-1, keepdim=True).cpu()
            y_true = data.y_dict['rec']

            out_lst = torch.flatten(y_true).tolist()
            pred_lst = torch.flatten(y_pred).tolist()
            out_df = pd.DataFrame({"y_pred": pred_lst, "y_true": out_lst})
            # print(y_pred, data.y_dict['paper'])
            # print(out_df)
            out_df.to_csv("GSaint_DBLP_conf_output.csv", index=None)
        else:
            print("start test")
            test()  # Test if inference on GPU succeeds.
            total_run_t = 0
            for run in range(args.runs):
                start_t = datetime.datetime.now()
                model.reset_parameters()
                for epoch in range(1, 1 + args.epochs):
                    loss = train(epoch)
                    torch.cuda.empty_cache()
                    result = test()
                    logger.add_result(run, result)
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%')
                logger.print_statistics(run)
                end_t = datetime.datetime.now()
                total_run_t = total_run_t + (end_t - start_t).total_seconds()
                print("model run ", run, " train time CPU=", end_t - start_t, " sec.")
                print(getrusage(RUSAGE_SELF))
            total_run_t = (total_run_t + 0.00001) / args.runs
            gsaint_end_t = datetime.datetime.now()
            Highest_Train, Highest_Valid, Final_Train, Final_Test = logger.print_statistics()
            model_trained_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
            dic_results[dataset_name]["init_ru_maxrss"] = init_ru_maxrss
            dic_results[dataset_name]["model_ru_maxrss"] = model_loaded_ru_maxrss
            dic_results[dataset_name]["model_trained_ru_maxrss"] = model_trained_ru_maxrss
            dic_results[dataset_name]["Highest_Train"] = Highest_Train.item()
            dic_results[dataset_name]["Highest_Valid"] = Highest_Valid.item()
            dic_results[dataset_name]["Final_Train"] = Final_Train.item()
            gsaint_Final_Test = Final_Train.item()
            dic_results[dataset_name]["Final_Test"] = Final_Test.item()
            dic_results[dataset_name]["runs_count"] = args.runs
            dic_results[dataset_name]["avg_train_time"] = total_run_t
            dic_results[dataset_name]["rgcn_total_time"] = (gsaint_end_t - gsaint_start_t).total_seconds()
            pd.DataFrame(dic_results).transpose().to_csv(
                "/shared_mnt/DBLP/OGBN_DBLP_GSAINT" + GA_dataset_name + "_GA_Times.csv", index=False)
            # shutil.rmtree("/shared_mnt/DBLP/" + dataset_name)
            # torch.save(model.state_dict(), "/shared_mnt/DBLP/" + dataset_name + "_DBLP_conf__GSAINT_QM.model")
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        print("dataset_name Exception")
    return gsaint_Final_Test


def init_graphSaintDataset(dataset_root_folder, dataset_name, class_count):
    print("dataset_name=", dataset_name)
    dataset = PygNodePropPredDataset_hsh(name=dataset_name, root=dataset_root_folder, numofClasses=str(class_count))
    return dataset


DBLP_OGBN_EdgeTypes_df = pd.read_csv("DBLP_OGBN_EdgeTypes.csv")
DBLP_OGBN_EdgeTypes_lst = DBLP_OGBN_EdgeTypes_df["EdgeType"].unique().tolist()


def genetic_algo(data, dataset_name, population_size, features_count, tol_level, top_number):
    def init_population(population_size, c, top_number):
        population = []
        for i in range(population_size):
            individual = [0] * c
            j = 0
            while (j < top_number):
                p = random.uniform(0, 1)
                position = random.randrange(c)
                if (p >= 0.5 and individual[position] == 0):
                    individual[position] = 1
                    j = j + 1

            # edge case if all genes are 0 then we will make any one gene as 1
            if (sum(individual) == 0):
                position = random.randrange(c)
                individual[position] = 1

            population.append(individual)
        print('population is ')
        print(population)
        print('------------------')
        return population

    # def calculate_fitness(features, target):
    #     model = MLPClassifier()
    #     scores = cross_val_score(model, features, target, scoring='f1_macro', n_jobs=-1,
    #                              cv=10)  # using f1_score as it is an imbalanced dataset
    #     print(scores.mean())
    #     return scores.mean()
    def calculate_fitness(to_remove_pedicates, to_remove_subject_object, to_keep_edge_idx_map, dataset, dataset_name):
        if not hasattr(calculate_fitness, "GA_Index"):
            calculate_fitness.GA_Index = 0  # it doesn't exist yet, so initialize it
        calculate_fitness.GA_Index += 1
        try:
            dataset = init_graphSaintDataset('/shared_mnt/DBLP/', dataset_name, 50)
            return graphSaint(to_remove_pedicates, to_remove_subject_object, to_keep_edge_idx_map,
                              calculate_fitness.GA_Index, dataset, dataset_name)
        except:
            return 0.1

    def get_filtred_edges_nodes(to_remove_edges_idx, to_keep_edges_idx):
        features_count = len(DBLP_OGBN_EdgeTypes_df)
        # remove_edges_idx_lst = [1, 5, 7, 13, 25]
        to_remove_pedicates = []
        to_remove_subject_object = []
        to_keep_edge_idx_map = []
        for idx in to_remove_edges_idx:
            to_remove_pedicates.append(DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["EdgeType"].values[0])
            if DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["ObjectType"].values[0].startswith("Object"):
                to_remove_subject_object.append(DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["ObjectType"].values[0])
        for idx in to_keep_edges_idx:
            to_keep_edge_idx_map.append(DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["edge_idx"].values[0])
        return to_remove_pedicates, to_remove_subject_object, to_keep_edge_idx_map

    def get_fitness(population, dataset):
        fitness_values = []
        for individual in population:
            i = 0
            to_remove_edges_idx = []
            to_keep_edges_idx = []
            print("individual=", individual)
            for idx, val in enumerate(individual):
                if (val == 0):
                    to_remove_edges_idx.append(idx)
                else:
                    to_keep_edges_idx.append(idx)

            to_remove_pedicates, to_remove_subject_object, to_keep_edge_idx_map = get_filtred_edges_nodes(
                to_remove_edges_idx, to_keep_edges_idx)
            individual_fitness = calculate_fitness(to_remove_pedicates, to_remove_subject_object, to_keep_edge_idx_map,
                                                   dataset, dataset_name)
            fitness_values.append(individual_fitness)

        return fitness_values

    def select_parents(population, fitness_values):
        parents = []
        total = sum(fitness_values)
        norm_fitness_values = [x / total for x in fitness_values]

        # find cumulative fitness values for roulette wheel selection
        cumulative_fitness = []
        start = 0
        for norm_value in norm_fitness_values:
            start += norm_value
            cumulative_fitness.append(start)

        population_size = len(population)
        for count in range(population_size):
            random_number = random.uniform(0, 1)
            individual_number = 0
            for score in cumulative_fitness:
                if (random_number <= score):
                    parents.append(population[individual_number])
                    break
                individual_number += 1
        return parents

    # high probability crossover
    def two_point_crossover(parents, probability):
        random.shuffle(parents)
        # count number of pairs for crossover
        no_of_pairs = round(len(parents) * probability / 2)
        chromosome_len = len(parents[0])
        crossover_population = []

        for num in range(no_of_pairs):
            length = len(parents)
            parent1_index = random.randrange(length)
            parent2_index = random.randrange(length)
            while (parent1_index == parent2_index):
                parent2_index = random.randrange(length)

            start = random.randrange(chromosome_len)
            end = random.randrange(chromosome_len)
            if (start > end):
                start, end = end, start

            parent1 = parents[parent1_index]
            parent2 = parents[parent2_index]
            child1 = parent1[0:start]
            child1.extend(parent2[start:end])
            child1.extend(parent1[end:])
            child2 = parent2[0:start]
            child2.extend(parent1[start:end])
            child2.extend(parent2[end:])
            parents.remove(parent1)
            parents.remove(parent2)
            crossover_population.append(child1)
            crossover_population.append(child2)

        # to append remaining parents which are not undergoing crossover process
        if (len(parents) > 0):
            for remaining_parents in parents:
                crossover_population.append(remaining_parents)

        return crossover_population

    # low probability mutation
    # mutation_probability is generally low to avoid a lot of randomness
    def mutation(crossover_population):
        # swapping of zero with one to retain no of features required
        for individual in crossover_population:
            index_1 = random.randrange(len(individual))
            index_2 = random.randrange(len(individual))
            while (index_2 == index_1 and individual[index_1] != individual[index_2]):
                index_2 = random.randrange(len(individual))

            # swapping the bits
            temp = individual[index_1]
            individual[index_1] = individual[index_2]
            individual[index_2] = temp

        return crossover_population

    def run(population_size, features_count, top_number, tol_level, dataset):
        c = features_count
        population = init_population(population_size, c, top_number)
        print("init_population done")
        fitness_values = get_fitness(population, dataset)
        print("get_fitness done")
        parents = select_parents(population, fitness_values)
        print("select_parents done")
        crossover_population = two_point_crossover(parents, 0.78)
        print("two_point_crossover done")
        population = crossover_population
        p = random.uniform(0, 1)
        if (p <= 0.001):
            mutated_population = mutation(crossover_population)
            population = mutated_population
        fitness_values = get_fitness(population, dataset)
        print("get_fitness for gen 0 done")
        variance_of_population = statistics.variance(fitness_values)
        print("variance for gen:0 is ", variance_of_population)
        gen = 1

        # repeating algorithm til stopping criterion is met
        while (variance_of_population > tol_level):
            parents = select_parents(population, fitness_values)
            print("select_parents for gen: ", gen, " done")
            crossover_population = two_point_crossover(parents, 0.78)
            print("crossover_population for gen: ", gen, " done")
            population = crossover_population
            p = random.uniform(0, 1)
            if (p <= 0.001):  # mutation prob here
                mutated_population = mutation(crossover_population)
                population = mutated_population
            fitness_values = get_fitness(population, data)
            print("get_fitness for gen: ", gen, " done")
            variance_of_population = statistics.variance(fitness_values)
            print("variance for gen:", gen, " is ", variance_of_population)
            gen += 1

        best_features = []
        best_f1_score = 0
        optimal_fitness = sum(fitness_values) / len(fitness_values)
        print('avg optimal_fitness is: ', optimal_fitness)
        for index, fitness_value in enumerate(fitness_values):
            error = abs((fitness_value - optimal_fitness) / optimal_fitness)
            if (error <= 0.01):
                best_features = population[index]
                best_f1_score = fitness_value
        print(best_features)
        return best_features, best_f1_score

    return run(population_size, features_count, top_number, tol_level, dataset)


if __name__ == '__main__':
    dataset_name = "dblp-2022-03-01_URI_Only_allPapers_Literals2Nodes_SY1900_EY2021_MAG03_AllEdgeTypes_PairsIdx_0_50Class"
    # dataset_name= "dblp-2022-03-01_URI_Only_allPapers_Literals2Nodes_SY2017_EY2021_MAG03_AllEdgeTypes_PairsIdx_0_50Class"
    dataset = init_graphSaintDataset('/home/hussein/Downloads/', dataset_name, 50)

    # keep_edges=["owl#sameAs","schema#archivedElectronicEdition","schema#doi","schema#yearOfEvent"]
    # keep_edges_idx=[]
    # for elem in keep_edges:
    #     keep_edges_idx.append(DBLP_OGBN_EdgeTypes_df[DBLP_OGBN_EdgeTypes_df["EdgeType"]==elem].index)
    # features_count = len(DBLP_OGBN_EdgeTypes_df)
    to_remove_edges_idx=[]
    individual = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1]
    # individual = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    individual=[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    for idx,val in enumerate(individual):
        if val ==0:
            to_remove_edges_idx.append(idx)
    # remove_edges_idx_lst = [1, 5, 7, 13, 25]
    to_remove_pedicates = []
    to_remove_subject_object = []
    for idx in to_remove_edges_idx:
        to_remove_pedicates.append(DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["EdgeType"].values[0])
        if DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["ObjectType"].values[0].startswith("Object"):
            to_remove_subject_object.append(DBLP_OGBN_EdgeTypes_df.iloc[[idx]]["ObjectType"].values[0])

    # print("un baised Sampler")
    # acc = graphSaint(to_remove_pedicates, to_remove_subject_object, [], 0, dataset, dataset_name,False)
    print("baised Sampler")
    acc=graphSaint(to_remove_pedicates, to_remove_subject_object,[], 0, dataset, dataset_name,True)


    # features_count=len(DBLP_OGBN_EdgeTypes_df)
    # top_features, best_f1_score = genetic_algo(dataset,dataset_name,10,features_count, 0.0005, 6)
    # # printing top features selected through genetic algorithm
    # i = 0
    # list_of_features = []
    # for i in range(len(top_features)):
    #     if (top_features[i] == 1):
    #         list_of_features.append(DBLP_OGBN_EdgeTypes_df.iloc[[i]]["EdgeType"].values[0])
    # print(top_features)
        # print(list_of_features)
    # print(best_f1_score)






