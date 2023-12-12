# %%
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.io import loadmat
import torch
import dgl
import random
import os
import time
import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
# from . import *
DATADIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "data/")


def featmap_gen(tmp_df=None):
    """
    Handle S-FFSD dataset and do some feature engineering
    :param tmp_df: the feature of input dataset
    """
    # time_span = [2, 5, 12, 20, 60, 120, 300, 600, 1500, 3600, 10800, 32400, 64800, 129600,
    #              259200]  # Increase in the number of time windows to increase the characteristics.
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df['trans_at_avg_{}'.format(
                tname)] = correct_data['Amount'].mean()
            new_df['trans_at_totl_{}'.format(
                tname)] = correct_data['Amount'].sum()
            new_df['trans_at_std_{}'.format(
                tname)] = correct_data['Amount'].std()
            new_df['trans_at_bias_{}'.format(
                tname)] = temp_amt - correct_data['Amount'].mean()
            new_df['trans_at_num_{}'.format(tname)] = len(correct_data)
            new_df['trans_target_num_{}'.format(tname)] = len(
                correct_data.Target.unique())
            new_df['trans_location_num_{}'.format(tname)] = len(
                correct_data.Location.unique())
            new_df['trans_type_num_{}'.format(tname)] = len(
                correct_data.Type.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)



def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def k_neighs(
    graph: dgl.DGLGraph,
    center_idx: int,
    k: int,
    where: str,
    choose_risk: bool = False,
    risk_label: int = 1
) -> torch.Tensor:
    """return indices of risk k-hop neighbors

    Args:
        graph (dgl.DGLGraph): dgl graph dataset
        center_idx (int): center node idx
        k (int): k-hop neighs
        where (str): {"predecessor", "successor"}
        risk_label (int, optional): value of fruad label. Defaults to 1.
    """
    target_idxs: torch.Tensor
    if k == 1:
        if where == "in":
            neigh_idxs = graph.predecessors(center_idx)
        elif where == "out":
            neigh_idxs = graph.successors(center_idx)

    elif k == 2:
        if where == "in":
            subg_in = dgl.khop_in_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_in.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            # delete center node itself
            neigh1s = graph.predecessors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]
        elif where == "out":
            subg_out = dgl.khop_out_subgraph(
                graph, center_idx, 2, store_ids=True)[0]
            neigh_idxs = subg_out.ndata[dgl.NID][subg_in.ndata[dgl.NID] != center_idx]
            neigh1s = graph.successors(center_idx)
            neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    neigh_labels = graph.ndata['label'][neigh_idxs]
    if choose_risk:
        target_idxs = neigh_idxs[neigh_labels == risk_label]
    else:
        target_idxs = neigh_idxs

    return target_idxs


def count_risk_neighs(
    graph: dgl.DGLGraph,
    risk_label: int = 1
) -> torch.Tensor:

    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)

    return torch.Tensor(ret)


def feat_map():
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),
            # edge_feat[neighs_1_of_center, 0].std().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),
            # edge_feat[neighs_2_of_center, 0].std().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),
            # edge_feat[neighs_1_of_center, 1].std().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),
            # edge_feat[neighs_2_of_center, 1].std().item(),
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree",
                  "1hop_riskstat", "2hop_riskstat"]

    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names


if __name__ == "__main__":

    set_seed(42)

    # # %%
    # """
    #     For S-FFSD dataset
    # """
    print(f"processing S-FFSD data...")
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []
    allt = []
    pair = ["Source", "Target", "Location", "Type"]
    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                        for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))
    cal_list = ["Source", "Target", "Location", "Type"]
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)
    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]
    g.ndata['label'] = torch.from_numpy(
        labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(
        feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + "graph-S-FFSD.bin", [g])

    # generate neighbor riskstat features
    for file_name in ['S-FFSD']:
        print(
            f"Generating neighbor risk-aware features for {file_name} dataset...")
        graph = dgl.load_graphs(DATADIR + "graph-" + file_name + ".bin")[0][0]
        graph: dgl.DGLGraph
        print(f"graph info: {graph}")

        edge_feat: torch.Tensor
        degree_feat = graph.in_degrees().unsqueeze_(1).float()
        risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()

        origin_feat_name = []
        edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
        origin_feat_name = ['degree', 'riskstat']

        features_neigh, feat_names = feat_map()
        # print(f"feature neigh: {features_neigh.shape}")

        features_neigh = torch.cat(
            (edge_feat, features_neigh), dim=1
        ).numpy()
        feat_names = origin_feat_name + feat_names
        features_neigh[np.isnan(features_neigh)] = 0.

        output_path = DATADIR + file_name + "_neigh_feat.csv"
        features_neigh = pd.DataFrame(features_neigh, columns=feat_names)
        scaler = StandardScaler()
        # features_neigh = np.log(features_neigh + 1)
        features_neigh = pd.DataFrame(scaler.fit_transform(
            features_neigh), columns=features_neigh.columns)

        features_neigh.to_csv(output_path, index=False)
