import  pandas as pd
import gzip
import os
import datetime
import shutil
from sklearn.metrics import precision_recall_fscore_support as score
def compress_gz(f_path,delete_file=True):
    f_in = open(f_path,'rb')
    f_out = gzip.open(f_path+".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    if delete_file:
        os.remove(f_path)
###################### Zip Folder to OGB Format
#zip -r mag_ComputerProgramming_papers_venue_QM3.zip mag_ComputerProgramming_papers_venue_QM3/ -i '*.gz'

if __name__ == '__main__':
    # csv_path="/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/ogb-mag.tsv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_AtmosphericSciences_papers_venue_QM1_BD_Query.csv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_AtmosphericSciences_papers_venue_QM1_Path_Query.csv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_AtmosphericSciences_papers_venue_QM1_BPath_Query.csv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_AtmosphericSciences_papers_venue_QM1.csv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_Diffraction_papers_venue_QM2.csv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_ComputerProgramming_papers_venue_QM3.csv"
    # csv_path = "/media/hussein/UbuntuData/GithubRepos/KG-EaaS/OGBN-Data/mag_QuantumElectrodynamics_papers_venue_QM4.csv"
    split_rel="http://mag.graph/has_year"
    target_rel = "http://mag.graph/has_venue"
    label_node="paper"
    fieldOfStudy_Coverage_df = pd.read_csv("/media/hussein/UbuntuData/OGBN_Datasets/ogbn_mag_fieldOfStudy_Coverage_top_10000.csv")
    fieldOfStudy_Coverage_df = fieldOfStudy_Coverage_df[fieldOfStudy_Coverage_df["do_train"] == 1].reset_index(drop=True)
    dic_results = {}
    root_path="/media/hussein/UbuntuData/OGBN_Datasets/"
    for i, row in fieldOfStudy_Coverage_df.iterrows():
        start_t = datetime.datetime.now()
        dataset_name=""
        if i>=0:
            dataset_name = "OBGN_MAG_Usecase_" + str(int(row["Q_idx"])) + "_" + str(str(row["topic"]).strip().replace(" ", "_").replace("/","_"))
            dic_results[dataset_name] = {}
            dic_results[dataset_name]["q_idx"] = int(row["Q_idx"])
            dic_results[dataset_name]["usecase"] = dataset_name
            print("dataset=", dataset_name)
            csv_path= root_path+ dataset_name + ".csv"
            split_by={"folder_name":"time","split_data_type":"int","train":2017,"valid":2018,"test":2019}
            if csv_path.endswith(".tsv"):
                g_tsv_df=pd.read_csv(csv_path,sep="\t")
            else:
                g_tsv_df = pd.read_csv(csv_path)
            try:
                g_tsv_df=g_tsv_df.rename(columns={"subject":"s","predicate":"p","object":"o"})
            except:
                print("g_tsv_df columns=",g_tsv_df.columns())
            ########################delete non target papers #####################
            lst_targets=g_tsv_df[g_tsv_df["p"]==target_rel]["s"].tolist()
            cites_df=g_tsv_df[g_tsv_df["p"]=="http://mag.graph/cites"]
            to_delete_papers=cites_df[~cites_df["o"].isin(lst_targets)]["o"].tolist()
            g_tsv_df = g_tsv_df[~g_tsv_df["o"].isin(to_delete_papers)]
            writes_df = g_tsv_df[g_tsv_df["p"] == "http://mag.graph/writes"]
            to_delete_papers = writes_df[~writes_df["o"].isin(lst_targets)]["o"].tolist()
            g_tsv_df=g_tsv_df[~g_tsv_df["o"].isin(to_delete_papers)]
            #####################################################################
            relations_lst=g_tsv_df["p"].unique().tolist()
            relations_lst.remove(split_rel)
            relations_lst.remove(target_rel)
            ################################write relations index ########################
            relations_df=pd.DataFrame(relations_lst, columns=["rel name"])
            relations_df["rel name"]=relations_df["rel name"].apply(lambda x: str(x).split("/")[-1])
            relations_df["rel idx"]=relations_df.index
            relations_df=relations_df[["rel idx","rel name"]]
            map_folder = root_path+dataset_name+"/mapping"
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)
            relations_df.to_csv(map_folder+"/relidx2relname.csv",index=None)
            compress_gz(map_folder+"/relidx2relname.csv")
            ############################### create label index ########################
            label_idx_df= pd.DataFrame(g_tsv_df[g_tsv_df["p"] == target_rel]["o"].apply(lambda x: str(x).strip()).unique().tolist(),columns=["label name"])
            try:
                label_idx_df["label name"] = label_idx_df["label name"].astype("int64")
                label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
            except:
                label_idx_df["label name"]=label_idx_df["label name"].astype("str")
                label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)

            label_idx_df["label idx"]=label_idx_df.index
            label_idx_df=label_idx_df[["label idx","label name"]]
            label_idx_df.to_csv(map_folder+"/labelidx2labelname.csv",index=None)
            compress_gz(map_folder+"/labelidx2labelname.csv")
            ###########################################prepare relations mapping#################################
            relations_entites_map={}
            relations_dic={}
            entites_dic={}
            for rel in relations_lst:
                relations_dic[rel]=g_tsv_df[g_tsv_df["p"]==rel].reset_index(drop=True)
                e1=str(relations_dic[rel]["s"][0]).split("/")
                e1=e1[len(e1)-2]
                e2 = str(relations_dic[rel]["o"][0]).split("/")
                e2 = e2[len(e2) - 2]
                relations_entites_map[rel]=(e1,rel,e2)
                if e1 in entites_dic:
                    entites_dic[e1]=entites_dic[e1].union(set(relations_dic[rel]["s"].apply(lambda x:str(x).split("/")[-1]).unique()))
                else:
                    entites_dic[e1] = set(relations_dic[rel]["s"].apply(lambda x:str(x).split("/")[-1]).unique())

                if e2 in entites_dic:
                    entites_dic[e2] = entites_dic[e2].union(set(relations_dic[rel]["o"].apply(lambda x:str(x).split("/")[-1]).unique()))
                else:
                    entites_dic[e2] = set(relations_dic[rel]["o"].apply(lambda x:str(x).split("/")[-1]).unique())

            ############################ write entites index #################################
            for key in list(entites_dic.keys()) :
                entites_dic[key]=pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype('int64').sort_values(by="ent name").reset_index(drop=True)
                entites_dic[key]=entites_dic[key].drop_duplicates()
                entites_dic[key]["ent idx"]=entites_dic[key].index
                entites_dic[key] = entites_dic[key][["ent idx","ent name"]]
                entites_dic[key+"_dic"]=pd.Series(entites_dic[key]["ent idx"].values,index=entites_dic[key]["ent name"]).to_dict()
                # print("key=",entites_dic[key+"_dic"])
                map_folder=root_path+dataset_name+"/mapping"
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)
                entites_dic[key].to_csv(map_folder+"/"+key+"_entidx2name.csv",index=None)
                compress_gz(map_folder+"/"+key+"_entidx2name.csv")
            #################### write nodes statistics ######################
            lst_node_has_feat= [list(filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
            lst_node_has_label=lst_node_has_feat.copy()
            lst_num_node_dict = lst_node_has_feat.copy()
            lst_has_feat = []
            lst_has_label=[]
            lst_num_node=[]

            for entity in lst_node_has_feat[0]:
                if str(entity)== str(label_node):
                    lst_has_label.append("True")
                    lst_has_feat.append("True")
                else:
                    lst_has_label.append("False")
                    lst_has_feat.append("False")

                # lst_has_feat.append("False")
                lst_num_node.append( len(entites_dic[entity+"_dic"]))

            lst_node_has_feat.append(lst_has_feat)
            lst_node_has_label.append(lst_has_label)
            lst_num_node_dict.append(lst_num_node)

            lst_relations=[]

            for k in list(relations_entites_map.keys()):
                (e1,rel,e2)=relations_entites_map[k]
                lst_relations.append([e1,str(rel).split("/")[-1],e2])

            map_folder = root_path+dataset_name + "/raw"
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)

            pd.DataFrame(lst_node_has_feat).to_csv(root_path+dataset_name + "/raw/nodetype-has-feat.csv", header=None, index=None)
            compress_gz(root_path+dataset_name + "/raw/nodetype-has-feat.csv")

            pd.DataFrame(lst_node_has_label).to_csv(root_path+dataset_name + "/raw/nodetype-has-label.csv", header=None, index=None)
            compress_gz(root_path+dataset_name + "/raw/nodetype-has-label.csv")

            pd.DataFrame(lst_num_node_dict).to_csv(root_path+dataset_name + "/raw/num-node-dict.csv", header=None, index=None)
            compress_gz(root_path+dataset_name + "/raw/num-node-dict.csv")

            pd.DataFrame(lst_relations).to_csv(root_path+dataset_name + "/raw/triplet-type-list.csv", header=None, index=None)
            compress_gz(root_path+dataset_name + "/raw/triplet-type-list.csv")
            ############################### create label relation index ######################
            label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
            labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel]
            label_type = str(labels_rel_df["s"].values[0]).split("/")
            label_type=label_type[len(label_type)-2]

            labels_rel_df["s_idx"]=labels_rel_df["s"].apply(lambda x: str(x).split("/")[-1])
            labels_rel_df["s_idx"] = labels_rel_df["s_idx"].astype("int64")
            labels_rel_df["s_idx"]=labels_rel_df["s_idx"].apply(lambda x: entites_dic[label_type+"_dic"][int(x)])
            labels_rel_df=labels_rel_df.sort_values(by=["s_idx"]).reset_index(drop=True)
            labels_rel_df["o_idx"]=labels_rel_df["o"].apply(lambda x: str(x).split("/")[-1])
            labels_rel_df["o_idx"]=labels_rel_df["o_idx"].apply(lambda x:label_idx_dic[int(x)])
            out_labels_df=labels_rel_df[["o_idx"]]
            map_folder = root_path+dataset_name + "/raw/node-label/"+label_type
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)
            out_labels_df.to_csv(map_folder+"/node-label.csv",header=None,index=None)
            compress_gz(map_folder+"/node-label.csv")
            ###########################################split parts (train/test/validate)#########################
            split_df = g_tsv_df[g_tsv_df["p"] == split_rel]
            label_type = str(split_df["s"].values[0]).split("/")
            label_type = label_type[len(label_type) - 2]
            try:
                split_df["s"] = split_df["s"].apply(lambda x: str(x).split("/")[-1]).astype("int64").apply(lambda x:entites_dic[label_type+"_dic"][x])
            except:
                split_df["s"] = split_df["s"].apply(lambda x: str(x).split("/")[-1]).astype("str").apply(lambda x: entites_dic[label_type + "_dic"][int(x)])
            split_df["o"]=split_df["o"].astype(split_by["split_data_type"])
            split_df=split_df.sort_values(by=["s"]).reset_index(drop=True)
            train_df = split_df[split_df["o"] <= split_by["train"]]["s"]
            valid_df = split_df[(split_df["o"] > split_by["train"]) & (split_df["o"] <= split_by["valid"])]["s"]
            test_df = split_df[ (split_df["o"] > split_by["valid"]) & (split_df["o"] <= split_by["test"])]["s"]

            map_folder = root_path+dataset_name + "/split/"+ split_by["folder_name"]+"/"+label_type
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)
            train_df.to_csv(map_folder + "/train.csv", index=None,header=None)
            compress_gz(map_folder + "/train.csv")
            valid_df.to_csv(map_folder + "/valid.csv", index=None, header=None)
            compress_gz(map_folder + "/valid.csv")
            test_df.to_csv(map_folder + "/test.csv", index=None, header=None)
            compress_gz(map_folder + "/test.csv")
            ###################### create nodetype-has-split.csv#####################
            lst_node_has_split=[ list(filter(lambda entity: str(entity).endswith("_dic")==False, list(entites_dic.keys())))]
            lst_has_split=[]
            for rel in lst_node_has_split[0]:
                if rel==label_type:
                    lst_has_split.append("True")
                else:
                    lst_has_split.append("False")
            lst_node_has_split.append(lst_has_split)
            pd.DataFrame(lst_node_has_split).to_csv(root_path+dataset_name + "/split/"+ split_by["folder_name"]+"/nodetype-has-split.csv",header=None,index=None)
            compress_gz(root_path+dataset_name + "/split/"+ split_by["folder_name"]+"/nodetype-has-split.csv")
            ############################ write entites relations  #################################
            idx=0
            for key in entites_dic["author_dic"].keys():
                # print(key, entites_dic["author_dic"][key])
                idx=idx+1
                if idx>10:
                    break;
            # print( list(entites_dic.keys()))
            for rel in relations_dic:
                e1,rel,e2=relations_entites_map[rel]
                relations_dic[rel]["s_idx"]=relations_dic[rel]["s"].apply(lambda x:entites_dic[e1+"_dic"][int(str(x).split("/")[-1])] ).astype("int64")
                relations_dic[rel]["o_idx"] = relations_dic[rel]["o"].apply(lambda x: entites_dic[e2 + "_dic"][int(str(x).split("/")[-1])]).astype("int64")
                relations_dic[rel]=relations_dic[rel].sort_values(by="s_idx").reset_index(drop=True)
                rel_out=relations_dic[rel].drop(columns=["s","p","o"])
                map_folder = root_path+dataset_name+"/raw/relations/"+e1+"___"+rel.split("/")[-1]+"___"+e2
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)
                rel_out.to_csv(map_folder + "/edge.csv", index=None, header=None)
                compress_gz(map_folder + "/edge.csv")
                ########## write relations num #################
                f = open(map_folder+"/num-edge-list.csv", "w")
                f.write(str(len(relations_dic[rel])))
                f.close()
                compress_gz(map_folder+"/num-edge-list.csv")
                ##################### write relations idx #######################
                rel_idx=relations_df[relations_df["rel name"]==rel.split("/")[-1]]["rel idx"].values[0]
                rel_out["rel_idx"]=rel_idx
                rel_idx_df=rel_out["rel_idx"]
                rel_idx_df.to_csv(map_folder+"/edge_reltype.csv",header=None,index=None)
                compress_gz(map_folder+"/edge_reltype.csv")
            #####################Zip Folder ###############3
            shutil.make_archive(root_path+dataset_name, 'zip', root_dir=root_path,base_dir=dataset_name)
            shutil.rmtree(root_path+dataset_name)
            end_t = datetime.datetime.now()
            print("csv_to_Hetrog_time=", end_t - start_t, " sec.")
            dic_results[dataset_name]["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()

        pd.DataFrame(dic_results).transpose().to_csv("/media/hussein/UbuntuData/OGBN_Datasets/OGBN_MAG_Uscases_CSVtoHetrog_times" + ".csv", index=False)
            # print(entites_dic)
