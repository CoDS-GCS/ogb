import logging
import os
import os.path as osp
from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
import numpy as np
import torch
import gzip
import pandas as pd
import rdflib as rdf
from torch_geometric.data import (Data, InMemoryDataset, download_url,extract_tar)
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling

class EntitiesMetaSampler(InMemoryDataset):
    r"""The relational entities networks "AIFB", "MUTAG", "BGS" and "AM" from
    the `"Modeling Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper.
    Training and test splits are given by node indices.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"AIFB"`,
            :obj:`"MUTAG"`, :obj:`"BGS"`, :obj:`"AM"`).
        hetero (bool, optional): If set to :obj:`True`, will save the dataset
            as a :class:`~torch_geometric.data.HeteroData` object.
            (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://data.dgl.ai/dataset/{}.tgz'

    def __init__(self, root: str=None, name: str=None, hetero: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,MaxNodeCount=40000,
                 Triples_df=None,labels_df=None,Train_df=None,Test_df=None):
        self.name = name
        self.relations_dict = {}
        self.nodes_dict = {}
        self.Triples_df=Triples_df
        self.Train_df=Train_df
        self.Test_df=Test_df
        self.labels_df=labels_df
        self.MaxNodeCount=MaxNodeCount
        if name is not None and name.lower() in ['aifb', 'am', 'mutag', 'bgs']:
            self.name = name.lower()
        self.hetero = hetero
        # assert self.name in ['aifb', 'am', 'mutag', 'bgs']
        if name:
            super().__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()

    @property
    def raw_dir(self) -> str:
        if self.root is None:
            return "."
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.root is None:
            return "."
        return osp.join(self.root, self.name, 'processed')

    @property
    def num_relations(self) -> int:
        return self.data.edge_type.max().item() + 1

    @property
    def num_classes(self) -> int:
        return self.data.train_y.max().item() + 1

    @property
    def raw_file_names(self) -> List[str]:
        return [
            ('stripped.nt.gz' if self.name is None else f'{self.name}_stripped.nt.gz'),
            'completeDataset.tsv',
            'trainingSet.tsv',
            'testSet.tsv',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'hetero_data.pt' if self.hetero else 'data.pt'

    def dfToRDFGraph(self,data_df):
        g = Graph()
        prefix = Namespace('https://sampledG/')
        g.bind('SG_prefix', prefix)
        S_URI = P_URI = O_URI = None
        for row in data_df.values:
            S_URI = URIRef(row[0])
            P_URI = URIRef(row[1])
            O_URI = URIRef(row[2])
            g.add((S_URI, P_URI, O_URI))
        return g
    def download(self):
        if self.name in ['aifb', 'am', 'mutag', 'bgs']:
            path = download_url(self.url.format(self.name), self.root)
            extract_tar(path, self.raw_dir)
            os.unlink(path)

    def process(self):

        if self.Triples_df is not None:
            g=self.dfToRDFGraph(self.Triples_df)
            freq = Counter(g.predicates())
            relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
            subjects = set(g.subjects())
            objects = set(g.objects())
            nodes = list(subjects.union(objects))
            try:
                sorted_nodes_ids = {}
                for elem in nodes:
                    sorted_nodes_ids[int(str(elem).split("/")[-1])] = elem
                temp_dict = sorted(list(sorted_nodes_ids.keys()))
                sorted_dict = {key: sorted_nodes_ids[key] for key in temp_dict}
                nodes = list(sorted_dict.values())
                last_elem = nodes[-1]
                nodes = nodes[0:self.MaxNodeCount - 1]
                nodes.append(last_elem)
            except:
                print("nodes ids are not int")

            N = len(nodes)
            R = 2 * len(relations)

            relations_dict = {rel: i for i, rel in enumerate(sorted(relations))}
            self.relations_dict = relations_dict
            nodes_dict = {node: i for i, node in enumerate(nodes)}
            self.nodes_dict = nodes_dict

            edges = []
            for s, p, o in g.triples((None, None, None)):
                try:
                    src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
                    edges.append([src, dst, 2 * rel])
                    edges.append([dst, src, 2 * rel + 1])
                except:
                    continue

            edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
            perm = (N * R * edges[0] + R * edges[1] + edges[2]).argsort()
            edges = edges[:, perm]
            edge_index, edge_type = edges[:2], edges[2]
            label_header = 'NScore'
            nodes_header = 'NodeID'

            labels_df = self.labels_df
            labels_set = set(labels_df[label_header].values.tolist())
            labels_dict = {float(lab): i for i, lab in
                           enumerate(sorted(list(labels_set)))}  ## sort to maintain class numerial orders
            nodes_dict = {np.unicode(key): val for key, val in nodes_dict.items()}

            train_labels_df = self.Train_df
            train_indices, train_labels = [], []
            for nod, lab in zip(train_labels_df[nodes_header].values,
                                train_labels_df[label_header].values):
                try:
                    train_indices.append(nodes_dict[nod])
                    train_labels.append(labels_dict[lab])  # use encoded labels
                    # train_labels.append(lab) # use actual values
                except:
                    continue

            train_idx = torch.tensor(train_indices, dtype=torch.long)
            train_y = torch.tensor(train_labels, dtype=torch.long)

            test_labels_df = self.Test_df
            test_indices, test_labels = [], []
            for nod, lab in zip(test_labels_df[nodes_header].values,
                                test_labels_df[label_header].values):
                try:
                    test_indices.append(nodes_dict[nod])
                    test_labels.append(labels_dict[lab])
                except:
                    continue

            test_idx = torch.tensor(test_indices, dtype=torch.long)
            test_y = torch.tensor(test_labels, dtype=torch.long)

            data = Data(edge_index=edge_index, edge_type=edge_type,
                        train_idx=train_idx, train_y=train_y, test_idx=test_idx,
                        test_y=test_y, num_nodes=N)

            if self.hetero:
                data = data.to_heterogeneous(node_type_names=['v'])
            self.data=data
            return data
        else:
            new_raw_paths = []
            if self.name in ['aifb', 'am', 'mutag', 'bgs']:
                graph_file, task_file, train_file, test_file = self.raw_paths
            else:
                for elem in self.raw_paths:
                    new_raw_paths.append(elem.split(self.name)[0] + self.name + "/raw/" + elem.split("/raw/")[-1])
                graph_file, task_file, train_file, test_file = new_raw_paths
                graph_file = graph_file.replace("_stripped.nt.gz", ".nt")

            with hide_stdout():
                g = rdf.Graph()
                if os.path.isfile(graph_file) and graph_file.endswith(".gz"):
                    with gzip.open(graph_file, 'rb') as f:
                        g.parse(file=f, format='nt')
                else:
                    with open(graph_file, 'rb') as f:
                        g.parse(file=f, format='nt')

            freq = Counter(g.predicates())

            relations = sorted(set(g.predicates()), key=lambda p: -freq.get(p, 0))
            subjects = set(g.subjects())
            objects = set(g.objects())
            nodes = list(subjects.union(objects))
            try:
                sorted_nodes_ids = {}
                for elem in nodes:
                    sorted_nodes_ids[int(str(elem).split("/")[-1])] = elem
                temp_dict = sorted(list(sorted_nodes_ids.keys()))
                sorted_dict = {key: sorted_nodes_ids[key] for key in temp_dict}
                nodes = list(sorted_dict.values())
                last_elem=nodes[-1]
                nodes=nodes[0:self.MaxNodeCount-1]
                nodes.append(last_elem)

            except:
                print("nodes ids are not int")

            N = len(nodes)
            R = 2 * len(relations)

            relations_dict = {rel: i for i, rel in enumerate(sorted(relations))}
            self.relations_dict = relations_dict
            nodes_dict = {node: i for i, node in enumerate(nodes)}
            self.nodes_dict = nodes_dict

            edges = []
            for s, p, o in g.triples((None, None, None)):
                try:
                    src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
                    edges.append([src, dst, 2 * rel])
                    edges.append([dst, src, 2 * rel + 1])
                except:
                    continue

            edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
            perm = (N * R * edges[0] + R * edges[1] + edges[2]).argsort()
            edges = edges[:, perm]

            edge_index, edge_type = edges[:2], edges[2]

            if self.name == 'am':
                label_header = 'label_cateogory'
                nodes_header = 'proxy'
            elif self.name == 'aifb':
                label_header = 'label_affiliation'
                nodes_header = 'person'
            elif self.name == 'mutag':
                label_header = 'label_mutagenic'
                nodes_header = 'bond'
            elif self.name == 'bgs':
                label_header = 'label_lithogenesis'
                nodes_header = 'rock'
            else:
                label_header = 'NScore'
                nodes_header = 'NodeID'

            labels_df = pd.read_csv(task_file, sep='\t')
            labels_set = set(labels_df[label_header].values.tolist())
            labels_dict = {float(lab): i for i, lab in
                           enumerate(sorted(list(labels_set)))}  ## sort to maintain class numerial orders
            pd.DataFrame(labels_dict.items(), columns=["label_val", "label_code"]).to_csv(
                task_file.replace(".tsv", "_label_codes.tsv"), index=None)
            nodes_dict = {np.unicode(key): val for key, val in nodes_dict.items()}

            train_labels_df = pd.read_csv(train_file, sep='\t')
            train_indices, train_labels = [], []
            for nod, lab in zip(train_labels_df[nodes_header].values,
                                train_labels_df[label_header].values):
                try:
                    train_indices.append(nodes_dict[nod])
                    train_labels.append(labels_dict[lab])  # use encoded labels
                    # train_labels.append(lab) # use actual values
                except:
                    continue

            train_idx = torch.tensor(train_indices, dtype=torch.long)
            train_y = torch.tensor(train_labels, dtype=torch.long)

            test_labels_df = pd.read_csv(test_file, sep='\t')
            test_indices, test_labels = [], []
            for nod, lab in zip(test_labels_df[nodes_header].values,
                                test_labels_df[label_header].values):
                try:
                    test_indices.append(nodes_dict[nod])
                    test_labels.append(labels_dict[lab])
                except:
                    continue

            test_idx = torch.tensor(test_indices, dtype=torch.long)
            test_y = torch.tensor(test_labels, dtype=torch.long)

            data = Data(edge_index=edge_index, edge_type=edge_type,
                        train_idx=train_idx, train_y=train_y, test_idx=test_idx,
                        test_y=test_y, num_nodes=N)

            if self.hetero:
                data = data.to_heterogeneous(node_type_names=['v'])
            torch.save(self.collate([data]), self.processed_paths[0])



    def __repr__(self) -> str:
        return f'{self.name.upper()}{self.__class__.__name__}()'


class hide_stdout(object):
    def __enter__(self):
        self.level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, *args):
        logging.getLogger().setLevel(self.level)
