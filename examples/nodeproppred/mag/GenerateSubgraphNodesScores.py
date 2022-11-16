import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import math
import pandas as pd #for handling csv and csv contents
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
from networkx import Graph as NXGraph
import matplotlib.pyplot as plt
import statistics
import collections
def generateSubgraphNodeScores(target_node_type,file_path,considerOversmoothing=True):
    # file_path="/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/DBLP_FG_DecodedSubgraph/DBLP_FG_GS_SubgraphNodes_2.csv"
    # target_node_type="CreativeWork"
    # file_path="/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/YAGO_PC_FM50_DecodedSubgraph/YAGO_PC_FM51_GS_SubgraphNodes_22.csv"
    file_name = os.path.basename(file_path)
    file_name = file_name.split(".")[0]
    #########read subgraph ##########
    SubgraphNodes_df = pd.read_csv(file_path)
    SubgraphNodes_df = SubgraphNodes_df[SubgraphNodes_df["Rel_type"].str.startswith("inv_") == False]
    if considerOversmoothing:
        normalized_df, all_labels_df, train, test, nt_df = generateSubgraphNodeScores_Oversmoothing_FromdDF(SubgraphNodes_df, 'rec',True)
    else:
        normalized_df,all_labels_df,train,test,nt_df=generateSubgraphNodeScoresFromdDF(SubgraphNodes_df,'rec',True)
    ######### Write Normalized Data ##########
    try:
        os.mkdir(file_path.replace(".csv", ""))
    except OSError as error:
        print(error)

    try:
        os.mkdir(file_path.replace(".csv", "") + "/raw")
    except OSError as error:
        print(error)
    ################ Write Dataset Files ##############
    normalized_df.to_csv(file_path.replace(".csv", "") + "/raw/" + file_name + "_normalizedSocres.tsv", header=None,sep="\t", index=None)
    all_labels_df.to_csv(file_path.replace(".csv", "") + "/raw/completeDataset.tsv", sep="\t", index=None)
    train.to_csv(file_path.replace(".csv", "") + "/raw/trainingSet.tsv", sep="\t", index=None)
    test.to_csv(file_path.replace(".csv", "") + "/raw/testSet.tsv", sep="\t", index=None)
    nt_df.to_csv(file_path.replace(".csv", "") + "/raw/" + file_name + ".nt", header=None, sep="\t", index=None)

def normalize(df,exclude):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name not in exclude:
            outliers_dic=dict((k, v) for k, v in df[feature_name].value_counts().to_dict().items() if v <= 10)
            if len(outliers_dic.keys())>0:
                # print(feature_name)
                min_key=min(outliers_dic.keys())
                df[feature_name]=df[feature_name].apply(lambda x: min_key if x in outliers_dic.keys() else x)
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
            result[feature_name]=result[feature_name].fillna(0)
            result[feature_name]=result[feature_name].apply(lambda x :int(math.floor(x*9)))
    return result

def generateSubgraphNodeScoresFromdDF(SubgraphNodes_df,target_node_type='rec',SetURI_Brackets=False):
    ######### list nodes/edges count  ##########
    node_types_count=len(set(SubgraphNodes_df["Src_Node_Type"].unique()).union(SubgraphNodes_df["Dest_Node_Type"].unique()))
    edge_types_count=len(SubgraphNodes_df["Rel_type"].unique())
    nodes=list(set(SubgraphNodes_df["Src_Node_ID"].unique()).union(SubgraphNodes_df["Dest_Node_ID"].unique()))
    nodes_count=len(set(SubgraphNodes_df["Src_Node_ID"].unique()).union(SubgraphNodes_df["Dest_Node_ID"].unique()))
    edges_count=len(SubgraphNodes_df)
    g_density=edges_count/(nodes_count*(nodes_count-1))
    ########## Calc Statictics ############
    out_edges_dic=SubgraphNodes_df["Src_Node_ID"].value_counts().to_dict()
    in_edges_dic=SubgraphNodes_df["Dest_Node_ID"].value_counts().to_dict()
    nodes_Score_df = pd.DataFrame(nodes,columns=["NodeID"])
    nodes_Score_df["OutEdges"]=nodes_Score_df.apply(lambda x: out_edges_dic[x.NodeID] if x.NodeID in out_edges_dic.keys() else 0  ,axis=1)
    nodes_Score_df["InEdges"]=nodes_Score_df.apply(lambda x: in_edges_dic[x.NodeID] if x.NodeID in in_edges_dic.keys() else 0  ,axis=1)
    ########## 1H Edges ##############
    TN_InEdges_1HC_dic=SubgraphNodes_df[SubgraphNodes_df["Src_Node_Type"]==target_node_type]["Dest_Node_ID"].value_counts().to_dict()
    TN_OutEdges_1HC_dic=SubgraphNodes_df[SubgraphNodes_df["Dest_Node_Type"]==target_node_type]["Src_Node_ID"].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_1HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_1HC_dic[x.NodeID] if x.NodeID in TN_InEdges_1HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_1HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_1HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_1HC_dic.keys() else 0  ,axis=1)
    ########## 2H Edges ##############
    TN_InEdges_2HC_dic=SubgraphNodes_df[SubgraphNodes_df["Src_Node_ID"].isin(list(TN_InEdges_1HC_dic.keys()))]["Dest_Node_ID"].value_counts().to_dict()
    TN_OutEdges_2HC_dic=SubgraphNodes_df[SubgraphNodes_df["Dest_Node_ID"].isin(list(TN_OutEdges_1HC_dic.keys()))]["Src_Node_ID"].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_2HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_2HC_dic[x.NodeID] if x.NodeID in TN_InEdges_2HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_2HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_2HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_2HC_dic.keys() else 0  ,axis=1)
    ########## 3H Edges ##############
    TN_InEdges_3HC_dic=SubgraphNodes_df[SubgraphNodes_df["Src_Node_ID"].isin(list(TN_InEdges_2HC_dic.keys()))]["Dest_Node_ID"].value_counts().to_dict()
    TN_OutEdges_3HC_dic=SubgraphNodes_df[SubgraphNodes_df["Dest_Node_ID"].isin(list(TN_OutEdges_2HC_dic.keys()))]["Src_Node_ID"].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_3HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_3HC_dic[x.NodeID] if x.NodeID in TN_InEdges_3HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_3HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_3HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_3HC_dic.keys() else 0  ,axis=1)
    ######### Is Target ################
    target_nodes_lst=list(set(SubgraphNodes_df[SubgraphNodes_df["Src_Node_Type"]==target_node_type]["Src_Node_ID"].unique()).union(SubgraphNodes_df[SubgraphNodes_df["Dest_Node_Type"]==target_node_type]["Dest_Node_ID"].unique()))
    nodes_Score_df["IsTarget"]=nodes_Score_df.apply(lambda x: 1 if x.NodeID in target_nodes_lst else 0  ,axis=1)
    ############ Normalize Scores ##############
    normalized_df=normalize(nodes_Score_df,exclude=["NodeID","IsTarget"])
    # normalized_df[normalized_df.InEdges>1]
    #############################Normalize NScore ###################################
    weightedCols=["IsTarget","TN_InEdges_1HC","TN_OutEdges_1HC",
                  "TN_InEdges_2HC","TN_OutEdges_2HC","InEdges","OutEdges","TN_InEdges_3HC",
                  "TN_OutEdges_3HC"]
    weight=10**9
    normalized_df["NScore"]=0
    for col in weightedCols:
        normalized_df['NScore']+= (normalized_df[col]*weight)
        weight/=10
        # print(weight)
    normalized_df['NScore']/=10**6

    all_labels_df=normalized_df[["NodeID","NScore"]]
    all_labels_df["NScore"]=all_labels_df["NScore"].apply(lambda x: int(math.ceil(x)))
    all_labels_df["NodeID"]=all_labels_df["NodeID"].apply(lambda x:"https://sampledG/NID/"+str(x)+"")
    all_labels_df.reset_index(inplace=True)
    all_labels_df = all_labels_df.rename(columns={'index': 'id'})
    ################ Train Test Split ################
    labels_frq_dic=all_labels_df["NScore"].value_counts().to_dict()
    non_unique_lst = [i for i in labels_frq_dic if labels_frq_dic[i]>1]
    all_labels_df = all_labels_df[all_labels_df["NScore"].isin(non_unique_lst)]
    train,test = train_test_split(all_labels_df, test_size=0.2,stratify=all_labels_df["NScore"])
    ################# Add Target Node Edge ###############
    nt_df=SubgraphNodes_df[["Src_Node_ID","Rel_ID","Dest_Node_ID"]]
    is_target_df= pd.DataFrame(target_nodes_lst,columns=["Src_Node_ID"])
    is_target_df["Rel_ID"]=nt_df["Rel_ID"].max()+1
    is_target_df["Dest_Node_ID"]=max(nt_df["Src_Node_ID"].max(),nt_df["Dest_Node_ID"].max())+1
    nt_df=nt_df.append(is_target_df)
    nt_df[nt_df["Rel_ID"]==nt_df["Rel_ID"].max()]
    ######################################################
    if SetURI_Brackets:
        nt_df["Src_Node_ID"]=nt_df["Src_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Dest_Node_ID"]=nt_df["Dest_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Rel_ID"]=nt_df["Rel_ID"].apply(lambda x:"<https://sampledG/RID/"+str(x)+">")
    else:
        nt_df["Src_Node_ID"] = nt_df["Src_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Dest_Node_ID"] = nt_df["Dest_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Rel_ID"] = nt_df["Rel_ID"].apply(lambda x: "https://sampledG/RID/" + str(x) )

    nt_df["eol"]="."
    return normalized_df,all_labels_df,train,test,nt_df
#add traget node edge as label for all traget nodes
def generateTargetLabedlSubgraph(SubgraphNodes_df,target_node_type='rec',SetURI_Brackets=False):
    ######### Is Target ################
    nodes = list(set(SubgraphNodes_df["Src_Node_ID"].unique()).union(SubgraphNodes_df["Dest_Node_ID"].unique()))
    nodes_Score_df = pd.DataFrame(nodes, columns=["NodeID"])
    target_nodes_lst=list(set(SubgraphNodes_df[SubgraphNodes_df["Src_Node_Type"]==target_node_type]["Src_Node_ID"].unique()).union(SubgraphNodes_df[SubgraphNodes_df["Dest_Node_Type"]==target_node_type]["Dest_Node_ID"].unique()))
    nodes_Score_df["IsTarget"]=nodes_Score_df.apply(lambda x: 1 if x.NodeID in target_nodes_lst else 0  ,axis=1)
    nodes_Score_df['NScore']=0
    all_labels_df=nodes_Score_df[["NodeID","NScore"]]
    all_labels_df["NScore"]=all_labels_df["NScore"].apply(lambda x: int(math.ceil(x)))
    all_labels_df["NodeID"]=all_labels_df["NodeID"].apply(lambda x:"https://sampledG/NID/"+str(x)+"")
    all_labels_df.reset_index(inplace=True)
    all_labels_df = all_labels_df.rename(columns={'index': 'id'})
    train,test = train_test_split(all_labels_df, test_size=0.2,stratify=all_labels_df["NScore"])
    ################# Add Target Node Edge ###############
    nt_df=SubgraphNodes_df[["Src_Node_ID","Rel_ID","Dest_Node_ID"]]
    is_target_df= pd.DataFrame(target_nodes_lst,columns=["Src_Node_ID"])
    is_target_df["Rel_ID"]=nt_df["Rel_ID"].max()+1
    is_target_df["Dest_Node_ID"]=max(nt_df["Src_Node_ID"].max(),nt_df["Dest_Node_ID"].max())+1
    nt_df=pd.concat([nt_df,is_target_df])
    nt_df[nt_df["Rel_ID"]==nt_df["Rel_ID"].max()]
    ##########################################
    if SetURI_Brackets:
        nt_df["Src_Node_ID"]=nt_df["Src_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Dest_Node_ID"]=nt_df["Dest_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Rel_ID"]=nt_df["Rel_ID"].apply(lambda x:"<https://sampledG/RID/"+str(x)+">")
    else:
        nt_df["Src_Node_ID"] = nt_df["Src_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Dest_Node_ID"] = nt_df["Dest_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Rel_ID"] = nt_df["Rel_ID"].apply(lambda x: "https://sampledG/RID/" + str(x) )

    nt_df["eol"]="."
    return all_labels_df,train,test,nt_df

def generateSubgraphNodeScoresFromTriplesdf(triples_df,all_nodes_ids,target_nodes_idx_lst):
    ######### list nodes/edges count  ##########
    edge_types_count=len(triples_df[1].unique())
    nodes=sorted(all_nodes_ids)
    nodes_count=len(nodes)
    edges_count=len(triples_df)
    g_density=edges_count/(nodes_count*(nodes_count-1))
    ########## Calc Statictics ############
    out_edges_dic=triples_df[0].value_counts().to_dict()
    in_edges_dic=triples_df[2].value_counts().to_dict()
    nodes_Score_df = pd.DataFrame(nodes,columns=["NodeID"])
    nodes_Score_df["OutEdges"]=nodes_Score_df.apply(lambda x: out_edges_dic[x.NodeID] if x.NodeID in out_edges_dic.keys() else 0  ,axis=1)
    nodes_Score_df["InEdges"]=nodes_Score_df.apply(lambda x: in_edges_dic[x.NodeID] if x.NodeID in in_edges_dic.keys() else 0  ,axis=1)
    ########## 1H Edges ##############
    TN_InEdges_1HC_dic=triples_df[triples_df[0].isin(target_nodes_idx_lst)][2].value_counts().to_dict()
    TN_OutEdges_1HC_dic=triples_df[triples_df[2].isin(target_nodes_idx_lst)][0].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_1HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_1HC_dic[x.NodeID] if x.NodeID in TN_InEdges_1HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_1HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_1HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_1HC_dic.keys() else 0  ,axis=1)
    ########## 2H Edges ##############
    TN_InEdges_2HC_dic=triples_df[triples_df[0].isin(list(TN_InEdges_1HC_dic.keys()))][2].value_counts().to_dict()
    TN_OutEdges_2HC_dic=triples_df[triples_df[2].isin(list(TN_OutEdges_1HC_dic.keys()))][0].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_2HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_2HC_dic[x.NodeID] if x.NodeID in TN_InEdges_2HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_2HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_2HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_2HC_dic.keys() else 0  ,axis=1)
    ########## 3H Edges ##############
    TN_InEdges_3HC_dic=triples_df[triples_df[0].isin(list(TN_InEdges_2HC_dic.keys()))][2].value_counts().to_dict()
    TN_OutEdges_3HC_dic=triples_df[triples_df[2].isin(list(TN_OutEdges_2HC_dic.keys()))][0].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_3HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_3HC_dic[x.NodeID] if x.NodeID in TN_InEdges_3HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_3HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_3HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_3HC_dic.keys() else 0  ,axis=1)
    ######### Is Target ################
    nodes_Score_df["IsTarget"]=nodes_Score_df.NodeID.apply(lambda x: 1 if x in target_nodes_idx_lst else 0)
    ############ Normalize Scores ##############
    normalized_df=normalize(nodes_Score_df,exclude=["NodeID","IsTarget"])
    # normalized_df[normalized_df.InEdges>1]
    #############################Normalize NScore ###################################
    weightedCols=["IsTarget","TN_InEdges_1HC","TN_OutEdges_1HC",
                  "TN_InEdges_2HC","TN_OutEdges_2HC","InEdges","OutEdges","TN_InEdges_3HC",
                  "TN_OutEdges_3HC"]
    weight=10**9
    normalized_df["NScore"]=0
    for col in weightedCols:
        normalized_df['NScore']+= (normalized_df[col]*weight)
        weight/=10
        # print(weight)
    normalized_df['NScore']/=10**6

    all_labels_df=normalized_df[["NodeID","NScore"]]
    all_labels_df["NScore"]=all_labels_df["NScore"].apply(lambda x: int(math.ceil(x)))
    ################ Train Test Split ################
    labels_frq_dic=all_labels_df["NScore"].value_counts().to_dict()
    non_unique_lst = [i for i in labels_frq_dic if labels_frq_dic[i]>1]
    temp_df = all_labels_df[all_labels_df["NScore"].isin(non_unique_lst)]
    train,test = train_test_split(temp_df, test_size=0.2,stratify=temp_df["NScore"])
    test_nodes_list=test["NodeID"].tolist()
    train_nodes_list =list(set(normalized_df["NodeID"].tolist())-set(test["NodeID"].tolist()))
    return normalized_df,all_labels_df,train_nodes_list,test_nodes_list

def generateSubgraphNodeScores_Oversmoothing_FromdDF(SubgraphNodes_df,target_node_type='rec',SetURI_Brackets=False):
    ######### list nodes/edges count  ##########
    node_types_count=len(set(SubgraphNodes_df["Src_Node_Type"].unique()).union(SubgraphNodes_df["Dest_Node_Type"].unique()))
    edge_types_count=len(SubgraphNodes_df["Rel_type"].unique())
    nodes=list(set(SubgraphNodes_df["Src_Node_ID"].unique()).union(SubgraphNodes_df["Dest_Node_ID"].unique()))
    nodes_count=len(set(SubgraphNodes_df["Src_Node_ID"].unique()).union(SubgraphNodes_df["Dest_Node_ID"].unique()))
    edges_count=len(SubgraphNodes_df)
    g_density=edges_count/(nodes_count*(nodes_count-1))
    ########## Calc Statictics ############
    out_edges_dic=SubgraphNodes_df["Src_Node_ID"].value_counts().to_dict()
    in_edges_dic=SubgraphNodes_df["Dest_Node_ID"].value_counts().to_dict()
    nodes_Score_df = pd.DataFrame(nodes,columns=["NodeID"])
    nodes_Score_df["OutEdges"]=nodes_Score_df.apply(lambda x: int((1/out_edges_dic[x.NodeID])*10) if x.NodeID in out_edges_dic.keys() else 0  ,axis=1)
    nodes_Score_df["InEdges"]=nodes_Score_df.apply(lambda x: int((1/in_edges_dic[x.NodeID])*10) if x.NodeID in in_edges_dic.keys() else 0  ,axis=1)
    ########## 1H Edges ##############
    TN_InEdges_1HC_dic=SubgraphNodes_df[SubgraphNodes_df["Src_Node_Type"]==target_node_type]["Dest_Node_ID"].value_counts().to_dict()
    TN_OutEdges_1HC_dic=SubgraphNodes_df[SubgraphNodes_df["Dest_Node_Type"]==target_node_type]["Src_Node_ID"].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_1HC"]=nodes_Score_df.apply(lambda x: int((1/TN_InEdges_1HC_dic[x.NodeID])*10) if x.NodeID in TN_InEdges_1HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_1HC"]=nodes_Score_df.apply(lambda x: int((1/TN_OutEdges_1HC_dic[x.NodeID])*10) if x.NodeID in TN_OutEdges_1HC_dic.keys() else 0  ,axis=1)
    ########## 2H Edges ##############
    TN_InEdges_2HC_dic=SubgraphNodes_df[SubgraphNodes_df["Src_Node_ID"].isin(list(TN_InEdges_1HC_dic.keys()))]["Dest_Node_ID"].value_counts().to_dict()
    TN_OutEdges_2HC_dic=SubgraphNodes_df[SubgraphNodes_df["Dest_Node_ID"].isin(list(TN_OutEdges_1HC_dic.keys()))]["Src_Node_ID"].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_2HC"]=nodes_Score_df.apply(lambda x: int((1/TN_InEdges_2HC_dic[x.NodeID])*10) if x.NodeID in TN_InEdges_2HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_2HC"]=nodes_Score_df.apply(lambda x: int((1/TN_OutEdges_2HC_dic[x.NodeID])*10) if x.NodeID in TN_OutEdges_2HC_dic.keys() else 0  ,axis=1)
    ########## 3H Edges ##############
    TN_InEdges_3HC_dic=SubgraphNodes_df[SubgraphNodes_df["Src_Node_ID"].isin(list(TN_InEdges_2HC_dic.keys()))]["Dest_Node_ID"].value_counts().to_dict()
    TN_OutEdges_3HC_dic=SubgraphNodes_df[SubgraphNodes_df["Dest_Node_ID"].isin(list(TN_OutEdges_2HC_dic.keys()))]["Src_Node_ID"].value_counts().to_dict()
    nodes_Score_df["TN_InEdges_3HC"]=nodes_Score_df.apply(lambda x: TN_InEdges_3HC_dic[x.NodeID] if x.NodeID in TN_InEdges_3HC_dic.keys() else 0  ,axis=1)
    nodes_Score_df["TN_OutEdges_3HC"]=nodes_Score_df.apply(lambda x: TN_OutEdges_3HC_dic[x.NodeID] if x.NodeID in TN_OutEdges_3HC_dic.keys() else 0  ,axis=1)
    ######### Is Target ################
    target_nodes_lst=list(set(SubgraphNodes_df[SubgraphNodes_df["Src_Node_Type"]==target_node_type]["Src_Node_ID"].unique()).union(SubgraphNodes_df[SubgraphNodes_df["Dest_Node_Type"]==target_node_type]["Dest_Node_ID"].unique()))
    nodes_Score_df["IsTarget"]=nodes_Score_df.apply(lambda x: 1 if x.NodeID in target_nodes_lst else 0  ,axis=1)
    ############ Normalize Scores ##############
    normalized_df=normalize(nodes_Score_df,exclude=["NodeID","IsTarget"])
    # normalized_df[normalized_df.InEdges>1]
    #############################Normalize NScore ###################################
    weightedCols=["IsTarget","TN_InEdges_1HC","TN_OutEdges_1HC",
                  "TN_InEdges_2HC","TN_OutEdges_2HC","InEdges","OutEdges","TN_InEdges_3HC",
                  "TN_OutEdges_3HC"]
    weight=10**9
    normalized_df["NScore"]=0
    for col in weightedCols:
        normalized_df['NScore']+= (normalized_df[col]*weight)
        weight/=10
        # print(weight)
    normalized_df['NScore']/=10**6

    all_labels_df=normalized_df[["NodeID","NScore"]]
    all_labels_df["NScore"]=all_labels_df["NScore"].apply(lambda x: int(math.ceil(x)))
    all_labels_df["NodeID"]=all_labels_df["NodeID"].apply(lambda x:"https://sampledG/NID/"+str(x)+"")
    all_labels_df.reset_index(inplace=True)
    all_labels_df = all_labels_df.rename(columns={'index': 'id'})
    ################ Train Test Split ################
    labels_frq_dic=all_labels_df["NScore"].value_counts().to_dict()
    non_unique_lst = [i for i in labels_frq_dic if labels_frq_dic[i]>1]
    all_labels_df = all_labels_df[all_labels_df["NScore"].isin(non_unique_lst)]
    train,test = train_test_split(all_labels_df, test_size=0.2,stratify=all_labels_df["NScore"])
    ################# Add Target Node Edge ###############
    nt_df=SubgraphNodes_df[["Src_Node_ID","Rel_ID","Dest_Node_ID"]]
    is_target_df= pd.DataFrame(target_nodes_lst,columns=["Src_Node_ID"])
    is_target_df["Rel_ID"]=nt_df["Rel_ID"].max()+1
    is_target_df["Dest_Node_ID"]=max(nt_df["Src_Node_ID"].max(),nt_df["Dest_Node_ID"].max())+1
    nt_df=nt_df.append(is_target_df)
    nt_df[nt_df["Rel_ID"]==nt_df["Rel_ID"].max()]
    ######################################################
    if SetURI_Brackets:
        nt_df["Src_Node_ID"]=nt_df["Src_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Dest_Node_ID"]=nt_df["Dest_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Rel_ID"]=nt_df["Rel_ID"].apply(lambda x:"<https://sampledG/RID/"+str(x)+">")
    else:
        nt_df["Src_Node_ID"] = nt_df["Src_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Dest_Node_ID"] = nt_df["Dest_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Rel_ID"] = nt_df["Rel_ID"].apply(lambda x: "https://sampledG/RID/" + str(x) )

    nt_df["eol"]="."
    return normalized_df,all_labels_df,train,test,nt_df

def generateSubgraphNodeScores_PPR_FromdDF(SubgraphNodes_df,target_node_type='rec',SetURI_Brackets=False):
    #################### build nx graph ##############
    rdf_g = Graph()
    # prefix_str='https://mag.org/'
    prefix_str = 'https://sampledG/'
    prefix = Namespace(prefix_str)
    rdf_g.bind('mag_prefix', prefix)
    S_URI = P_URI = O_URI = None
    for index, row in SubgraphNodes_df.iterrows():
        S_URI = URIRef(prefix_str + row[0] + "/" + str(row[1]))
        P_URI = URIRef(prefix_str + str(row[3]))
        O_URI = URIRef(prefix_str + row[4] + "/" + str(row[5]))
        rdf_g.add((S_URI, P_URI, O_URI))
    # print(g.serialize(format='turtle').decode('UTF-8'))
    # g.serialize('OGB_MAG_DecodedSubgraph_1.nt',format='n3')
    nx_g = rdflib_to_networkx_graph(rdf_g)
    nodes_lst = list(nx_g.nodes())
    dic_per = {}
    for n in nodes_lst:
        if "/" + target_node_type + "/" in str(n):
            dic_per[n] = 1
    ppr1 = nx.pagerank(nx_g, personalization=dic_per)
    scores_dic={}
    for k, v in ppr1.items():
        scores_dic["https://sampledG/NID/"+(str(k).split("/")[-1])]=int(v*10e6)
    all_labels_df=pd.DataFrame(list(scores_dic.items()), columns=['NodeID', 'NScore'])
    all_labels_df.reset_index(inplace=True)
    all_labels_df = all_labels_df.rename(columns={'index': 'id'})
    ################ Train Test Split ################
    labels_frq_dic=all_labels_df["NScore"].value_counts().to_dict()
    non_unique_lst = [i for i in labels_frq_dic if labels_frq_dic[i]>1]
    all_labels_df = all_labels_df[all_labels_df["NScore"].isin(non_unique_lst)]
    train,test = train_test_split(all_labels_df, test_size=0.2,stratify=all_labels_df["NScore"])
    ################# Add Target Node Edge ###############
    nt_df=SubgraphNodes_df[["Src_Node_ID","Rel_ID","Dest_Node_ID"]]
    target_nodes_lst = list(set(SubgraphNodes_df[SubgraphNodes_df["Src_Node_Type"] == target_node_type]["Src_Node_ID"].unique()).union(
            SubgraphNodes_df[SubgraphNodes_df["Dest_Node_Type"] == target_node_type]["Dest_Node_ID"].unique()))
    is_target_df= pd.DataFrame(target_nodes_lst,columns=["Src_Node_ID"])
    is_target_df["Rel_ID"]=nt_df["Rel_ID"].max()+1
    is_target_df["Dest_Node_ID"]=max(nt_df["Src_Node_ID"].max(),nt_df["Dest_Node_ID"].max())+1
    nt_df=nt_df.append(is_target_df)
    nt_df[nt_df["Rel_ID"]==nt_df["Rel_ID"].max()]
    ######################################################
    if SetURI_Brackets:
        nt_df["Src_Node_ID"]=nt_df["Src_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Dest_Node_ID"]=nt_df["Dest_Node_ID"].apply(lambda x:"<https://sampledG/NID/"+str(x)+">")
        nt_df["Rel_ID"]=nt_df["Rel_ID"].apply(lambda x:"<https://sampledG/RID/"+str(x)+">")
    else:
        nt_df["Src_Node_ID"] = nt_df["Src_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Dest_Node_ID"] = nt_df["Dest_Node_ID"].apply(lambda x: "https://sampledG/NID/" + str(x) )
        nt_df["Rel_ID"] = nt_df["Rel_ID"].apply(lambda x: "https://sampledG/RID/" + str(x) )

    nt_df["eol"]="."
    return all_labels_df,train,test,nt_df



target_node_type="rec"
# file_path="/media/hussein/UbuntuData/GithubRepos/ogb_cods/examples/nodeproppred/mag/DBLP_FG_DecodedSubgraph/DBLP_FG_GS_SubgraphNodes_7.csv"
# for i in range (8,20):
#     print(i)
#     generateSubgraphNodeScores(target_node_type,file_path.replace("_7.csv","_"+str(i)+".csv"))