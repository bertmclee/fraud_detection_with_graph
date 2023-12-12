import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
cat_features = ["Target",
                "Type",
                "Location"]


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base
		:param bias: the encoding bias
        """
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


# class TransEmbedding(nn.Module):

#     def __init__(self, df=None, device='cpu', dropout=0.2, feats_dim=82, cat_features=None, source_emb=None, target_emb=None, update_emb=False):
#         """
#         Initialize the attribute embedding and feature learning compoent

#         :param df: the feature
#                 :param device: where to train model
#                 :param dropout: the dropout rate
#                 :param in_feat: the shape of input feature in dimension 1
#                 :param cat_feature: category features
#         """
#         super(TransEmbedding, self).__init__()
#         self.time_pe = PosEncoding(dim=feats_dim, device=device, base=100)
#         #time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
#         self.label_table = nn.Embedding(3, feats_dim, padding_idx=2).to(device)
#         self.time_emb = None

#         # TODO
#         if edge_emb is None:
#             self.forward_method = "original"
#             self.emb_dict = nn.ModuleDict({col: nn.Embedding(max(df[col].unique())+1, feats_dim).to(device) 
#                                         for col in cat_features if col not in {"Labels", "Time"}})
#         else:
#             if not update_emb:
#                 self.forward_method = "no_update_emb"
#                 self.emb_dict = nn.ModuleDict({col: nn.Embedding(max(df[col].unique())+1, feats_dim).to(device) 
#                                         for col in cat_features if col not in {"Labels", "Time", "Source", "Target"}})
#             else:
#                 self.forward_method = "update_emb"
#                 self.emb_dict = ...

#         self.label_emb = None
#         self.cat_features = cat_features
#         self.forward_mlp = nn.ModuleList([nn.Linear(feats_dim, feats_dim) for i in range(len(cat_features))])
#         self.dropout = nn.Dropout(dropout)

#     # TODO
#     def forward(self, df):
#         if self.forward_method == "original":
#             support = {col: self.emb_dict[col](df[col]) 
#                     for col in self.cat_features if col not in {"Labels", "Time"}}
#             output = 0
#             for i, k in enumerate(support.keys()):
#                 support[k] = self.dropout(support[k])
#                 support[k] = self.forward_mlp[i](support[k])
#                 output = output + support[k]
#             return output
#         else:
#             if self.forward_method == "no_update_emb":
#                 support = {col: self.emb_dict[col](df[col]) 
#                         for col in self.cat_features if col not in {"Labels", "Time", "Source", "Target"}}
#                 output = 0
#                 for i, k in enumerate(support.keys()):
#                     support[k] = self.dropout(support[k])
#                     support[k] = self.forward_mlp[i](support[k])
#                     output = output + support[k]
#                 output = mlp(concat([output, source_emb, target_emb]))
#                 return output
#             else:   
#                 support = {col: self.emb_dict[col](df[col]) 
#                         for col in self.cat_features if col not in {"Labels", "Time"}}
#                 output = 0
#                 for i, k in enumerate(support.keys()):
#                     support[k] = self.dropout(support[k])
#                     support[k] = self.forward_mlp[i](support[k])
#                 output = concat(output, support[k])
#                 return mlp(output)


# class TransEmbedding(nn.Module):
#     def __init__(self, df=None, device='cpu', dropout=0.2, node_emb_dim=82, feats_dim=82, cat_features=None, source_emb=None, target_emb=None, update_pretrained_emb=True):
#         super(TransEmbedding, self).__init__()
#         self.cat_features = cat_features
#         self.dropout = nn.Dropout(dropout)
#         self.update_pretrained_emb = update_pretrained_emb

#         # Handle pre-trained source and target embeddings
#         self.source_emb = nn.Parameter(source_emb, requires_grad=update_pretrained_emb) if source_emb is not None else None
#         self.target_emb = nn.Parameter(target_emb, requires_grad=update_pretrained_emb) if target_emb is not None else None

#         # Embedding dictionaries for categorical features
#         self.emb_dict = nn.ModuleDict({
#             col: nn.Embedding(max(df[col].unique())+1, node_emb_dim).to(device)
#             for col in cat_features if col not in {"Labels", "Time", "Source", "Target"}
#         })

#         # MLP for combined embeddings
#         input_dim = node_emb_dim * (len(cat_features)-2)
#         if source_emb is not None:
#             input_dim += source_emb.shape[1] * 2  # Source and target embeddings
#         # print(f'node_emb_dim: {node_emb_dim}, cat_features: {len(cat_features)}, source_emb: {source_emb.shape[1]}, input_dim: {input_dim}')
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, node_emb_dim * 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(node_emb_dim * 2, feats_dim),
#             nn.LayerNorm(feats_dim)  # Normalization for final embeddings
#         )
#         # Layer normalization for embeddings
#         self.emb_norm = nn.LayerNorm(input_dim)

#     def forward(self, df):
#         cat_embeddings = [self.emb_dict[col](df[col]) for col in self.cat_features if col not in {"Labels", "Time", "Source", "Target"}]
#         cat_embeddings = torch.cat(cat_embeddings, dim=1) if cat_embeddings else torch.tensor([]).to(df.device)

#         if self.source_emb is not None and self.target_emb is not None:
#             source_idx = df["Source"].long()
#             target_idx = df["Target"].long()
#             source_emb = self.source_emb[source_idx]
#             target_emb = self.target_emb[target_idx]
#             combined_embeddings = torch.cat((cat_embeddings, source_emb, target_emb), dim=1)
#         else:
#             combined_embeddings = cat_embeddings

#         # Normalize combined embeddings
#         norm_embeddings = self.emb_norm(combined_embeddings)
#         # Pass through MLP
#         return self.mlp(norm_embeddings)


import torch
import torch.nn as nn

class TransEmbedding(nn.Module):
    def __init__(self, df=None, device='cpu', dropout=0.2, node_emb_dim=82, feats_dim=82, cat_features=None, source_emb=None, target_emb=None, update_pretrained_emb=True):
        super(TransEmbedding, self).__init__()
        self.cat_features = cat_features
        self.dropout = nn.Dropout(dropout)
        self.update_pretrained_emb = update_pretrained_emb

        # Handle pre-trained source and target embeddings
        self.source_emb = nn.Parameter(source_emb, requires_grad=update_pretrained_emb) if source_emb is not None else None
        self.target_emb = nn.Parameter(target_emb, requires_grad=update_pretrained_emb) if target_emb is not None else None

        # Embedding dictionaries for categorical features
        self.emb_dict = nn.ModuleDict({
            col: nn.Embedding(max(df[col].unique())+1, node_emb_dim).to(device)
            for col in cat_features if col not in {"Labels", "Time", "Source", "Target"}
        })

        # Calculate total input dimension
        input_dim = node_emb_dim * (len(cat_features)-2)
        if source_emb is not None:
            input_dim += source_emb.shape[1]
        if target_emb is not None:
            input_dim += target_emb.shape[1]

        # Layer normalization for embeddings
        self.emb_norm = nn.LayerNorm(input_dim)

        # MLP for combined embeddings
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * feats_dim),  # Expanding dimension
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * feats_dim, feats_dim),  # Reducing to desired dimension
            nn.LayerNorm(feats_dim)  # Normalization for final embeddings
        )

    def forward(self, df):
        # Gather embeddings for categorical features
        cat_embeddings = [self.emb_dict[col](df[col]) for col in self.cat_features if col not in {"Labels", "Time", "Source", "Target"}]
        cat_embeddings = torch.cat(cat_embeddings, dim=1) if cat_embeddings else torch.tensor([]).to(df.device)

        # Handle source and target embeddings
        if self.source_emb is not None and self.target_emb is not None:
            source_idx = df["Source"].long()
            target_idx = df["Target"].long()
            source_emb = self.source_emb[source_idx]
            target_emb = self.target_emb[target_idx]
            combined_embeddings = torch.cat((cat_embeddings, source_emb, target_emb), dim=1)
        else:
            combined_embeddings = cat_embeddings

        # Normalize combined embeddings
        norm_embeddings = self.emb_norm(combined_embeddings)

        # Pass through MLP
        return self.mlp(norm_embeddings)



# class TransEmbedding(nn.Module):

#     def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=82, cat_features=None):
#         """
#         Initialize the attribute embedding and feature learning compoent

#         :param df: the feature
#                 :param device: where to train model
#                 :param dropout: the dropout rate
#                 :param in_feat: the shape of input feature in dimension 1
#                 :param cat_feature: category features
#         """
#         super(TransEmbedding, self).__init__()
#         self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
#         #time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
#         self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
#         ))+1, in_feats).to(device) for col in cat_features if col not in {"Labels", "Time"}})
#         self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
#         self.time_emb = None
#         self.emb_dict = None
#         self.label_emb = None
#         self.cat_features = cat_features
#         self.forward_mlp = nn.ModuleList(
#             [nn.Linear(in_feats, in_feats) for i in range(len(cat_features))])
#         self.dropout = nn.Dropout(dropout)

#     def forward_emb(self, df):
#         if self.emb_dict is None:
#             self.emb_dict = self.cat_table
#         # print(self.emb_dict)
#         # print(df['trans_md'])
#         support = {col: self.emb_dict[col](
#             df[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
#         #self.time_emb = self.time_pe(torch.sin(torch.tensor(df['time_span'])/86400*torch.pi))
#         #support['time_span'] = self.time_emb
#         #support['labels'] = self.label_table(df['labels'])
#         return support

#     def forward(self, df):
#         support = self.forward_emb(df)
#         output = 0
#         for i, k in enumerate(support.keys()):
#             # if k =='time_span':
#             #    print(df[k].shape)
#             support[k] = self.dropout(support[k])
#             support[k] = self.forward_mlp[i](support[k])
#             output = output + support[k]
#         return output


class TransformerConv(nn.Module):

    def __init__(self,
                 feats_dim,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention 
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature 
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function   
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(feats_dim)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)

        #self.feat_dropout = nn.Dropout(p=feat_drop)
        #self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param get_attention: whether to get attention
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Step 0. q, k, v
        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats**0.5)

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats*self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class GraphAttnModel(nn.Module):
    def __init__(self,
                 feats_dim,
                 node_emb_dim,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 nei_features=None,
                 source_emb=None,
                 target_emb=None,
                 device='cpu'):
        """
        Initialize the GTAN-GNN model
        :param feats_dim: the shape of input feature
                :param hidden_dim: model hidden layer dimension
                :param n_layers: the number of GTAN layers
                :param n_classes: the number of classification
                :param heads: the number of multi-head attention 
                :param activation: the type of activation function
                :param skip_feat: whether to skip some feature
                :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
                :param post_proc: whether to use post processing
                :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
                :param ref_df: whether to refer other node features
                :param cat_features: category features
                :param nei_features: neighborhood statistic features
                :param source_emb: pre-trained embedding of source node
                :param target_emb: pre-trained embedding of target node
        :param device: where to train model
        """

        super(GraphAttnModel, self).__init__()
        self.feats_dim = feats_dim
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        #self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.dropout = nn.Dropout(dropout)
        # self.pn = PairNorm(mode=pairnorm)

        # TODO GGTAN
        if n2v_feat:
            # self.n2v_mlp = TransEmbedding(ref_df, device=device, feats_dim=feats_dim, cat_features=cat_features)
            self.n2v_mlp = TransEmbedding(ref_df, device=device, feats_dim = feats_dim, node_emb_dim = node_emb_dim,
                                          cat_features=cat_features, source_emb=source_emb, 
                                          target_emb=target_emb, update_pretrained_emb=False)
            
        else:
            self.n2v_mlp = lambda x: x

        # MLP for combined embeddings
        # input_dim = feats_dim * (len(cat_features)-2)
        # if source_emb is not None:
        #     input_dim += source_emb.shape[1] * 2  # Source and target embeddings
        # print(f'feats_dim: {feats_dim}, cat_features: {len(cat_features)}, source_emb: {source_emb.shape[1]}, input_dim: {input_dim}')

        input_dim = feats_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, feats_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(feats_dim * 2, feats_dim)
        )

        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(n_classes+1, feats_dim, padding_idx=n_classes))
        self.layers.append(nn.Linear(self.feats_dim, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Linear(self.feats_dim, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], feats_dim)
                                         ))

        # build multiple layers
        self.layers.append(TransformerConv(feats_dim=self.feats_dim,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(feats_dim=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes))
        
        # Normalization layers for features
        self.feature_batch_norm = nn.BatchNorm1d(feats_dim)
        self.feature_layer_norm = nn.LayerNorm(feats_dim)

    def forward(self, blocks, labels, features, n2v_feat=None):
        """
        :param blocks: train blocks
        :param features: train features  (|input|, feta_dim)
        :param labels: train labels (|input|, )
        :param n2v_feat: whether to use n2v features 
        """
        # print(f'features: {features.shape}')
        # print(n2v_feat['Source'].shape)
        # print(f'n2v_feat: {n2v_feat.shape}')
        # Batch and layer normalization for numerical features
        features = self.feature_batch_norm(features)
        features = self.feature_layer_norm(features)

        if n2v_feat is None:
            h = features
        else:
            # print(n2v_feat)
            h = self.n2v_mlp(n2v_feat)
            # TO DO GGTAN
            # print(f'features: {features.shape}, h: {h.shape}')
            # h = features + h
            h = torch.cat((features, h), dim=1)
            h = self.mlp(h)
            h = self.feature_layer_norm(h)
        
        label_embed = self.input_drop(self.layers[0](labels))
        # label_embed = self.layers[1](h) + self.layers[2](label_embed)
        label_embed = self.feature_layer_norm(label_embed)
        label_embed = torch.cat((h, label_embed), dim=1)
        # label_embed = self.layers[3](label_embed)
        label_embed = self.mlp(label_embed)
        label_embed = self.feature_layer_norm(label_embed)

        h = h + label_embed  # residual

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l+4](blocks[l], h))

        logits = self.layers[-1](h)

        return logits


