import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
# from torch_geometric.datasets import Entities
from examples.nodeproppred.mag.EntitiesMetaSampler import EntitiesMetaSampler as Entities
from torch_geometric.nn import RGCNConv
from sklearn.metrics import mean_squared_error
import operator
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
def test(model,data):
    model.eval()
    out = model(data.edge_index, data.edge_type)
    # pred = out[data.test_idx].max(1)[1]
    # acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    acc=mean_squared_error(data.test_y,torch.flatten(out[data.test_idx]).detach())
    return acc,torch.flatten(out[data.test_idx]).detach(),data.test_y

model,data=None,None
def train(path,name,nepochs=100):
    datasets=[]
    for ds in range(5,16):
        dataset = Entities(path+name+str(ds), name+str(ds), MaxNodeCount=40000)
        # print(len(dataset.nodes_dict))
        # print(len(dataset.relations_dict))
        # print(dataset)
        data = dataset[0]
        print(data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data=data.to(device)
        datasets.append(data)
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

def main():
    ######################################
    name = 'DBLP_FG_GS_SubgraphNodes_'
    path = '/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/DBLP_FG_DecodedSubgraph/'
    # model=train(path,name,20)
    # saveModel(model,path,"DBLP_FG_GS_SubgraphNodes_MetaSampler")
    #####################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model=loadModel(model,path,"DBLP_FG_GS_SubgraphNodes_MetaSampler")
    total_acc=0
    for i in range(6,20):
        dataset = Entities(path+name+str(i), name+str(i), MaxNodeCount=40000)
        data = dataset[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data=data.to(device)
        keys,dict=getTopKNodes(model,data,75)
        test_acc, out, true = test(model,data)
        total_acc+=test_acc
        print("test_acc", test_acc)
    print("avg=",total_acc/13)
    # exportModelOutput(model,data,path,name)
