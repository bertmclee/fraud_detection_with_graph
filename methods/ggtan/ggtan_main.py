import os
import time
import pickle
import numpy as np
import pandas as pd
import scipy
import torch
import dgl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.sparse.linalg import eigsh, eigs
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer, StandardScaler
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_scipy_sparse_matrix
from tqdm import tqdm

from .ggtan_model import GraphAttnModel
from . import *



def ggtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features, source_emb, target_emb, gtan_use_pese=False):
    """
    Main function for the GTAN model training and evaluation.

    Args:
        feat_df (DataFrame): Feature dataframe.
        graph (DGLGraph): Graph object.
        train_idx (list): Indices for training data.
        test_idx (list): Indices for testing data.
        labels (Series): Target labels.
        args (dict): Configuration arguments including device, number of folds, etc.
        cat_features (list): List of categorical features.
        source_emb (Tensor): Source embeddings tensor.
        target_emb (Tensor): Target embeddings tensor.
        gtan_use_pese (bool): Flag to use positional and structural encoding.
    """

    device = args['device']
    graph = graph.to(device)

    if gtan_use_pese == True:
        dim_lpe = 32  # Dimension of Laplacian PE
        dim_rwse = 32  # Dimension of Random Walk SE
        walk_length = 5  # Length of random walks for RWSE

        # Compute the encodings
        lpe = laplacian_positional_encoding_sparse(graph, dim_lpe)
        rwse = random_walk_structural_encoding(graph, walk_length, dim_rwse)

        # Convert to Pandas DataFrame
        lpe_df = pd.DataFrame(lpe, columns=[f'LPE_{i}' for i in range(lpe.shape[1])])
        rwse_df = pd.DataFrame(rwse, columns=[f'RWSE_{i}' for i in range(rwse.shape[1])])

        # Concatenate with feat_data
        feat_df = pd.concat([feat_df, lpe_df, rwse_df], axis=1)

    oof_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    test_predictions = torch.from_numpy(np.zeros([len(feat_df), 2])).float().to(device)
    kfold = StratifiedKFold(n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {col: torch.from_numpy(feat_df[col].values).long().to(device) for col in cat_features}

    y = labels
    labels = torch.from_numpy(y.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind, val_ind = torch.from_numpy(np.array(train_idx)[trn_idx]).long().to(device), \
                            torch.from_numpy(np.array(train_idx)[val_idx]).long().to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = DataLoader(graph,
                                          trn_ind,
                                          train_sampler,
                                          device=device,
                                          use_ddp=False,
                                          batch_size=args['batch_size'],
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=0
                                          )
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = DataLoader(graph,
                                        val_ind,
                                        val_sampler,
                                        use_ddp=False,
                                        device=device,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        )
        # TODO
        model = GraphAttnModel(feats_dim=feat_df.shape[1],
                               node_emb_dim=source_emb.shape[1],
                               hidden_dim=args['hid_dim']//4,
                               n_classes=2,
                               heads=[4]*args['n_layers'],  # [4,4,4]
                               activation=nn.PReLU(),
                               n_layers=args['n_layers'],
                               drop=args['dropout'],
                               device=device,
                               gated=args['gated'],
                               ref_df=feat_df.iloc[train_idx],
                               cat_features=cat_feat,
                               source_emb=source_emb,
                               target_emb=target_emb
                               ).to(device)
        lr = args['lr'] * np.sqrt(args['batch_size']/1024)  # 0.00075
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[4000, 12000], gamma=0.3)

        earlystoper = early_stopper(
            patience=args['early_stopping'], verbose=True)
        start_epoch, max_epochs = 0, 2000
        for epoch in range(start_epoch, args['max_epochs']):
            train_loss_list = []
            # train_acc_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_num_inputs, batch_cat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device)
                # (|input|, feat_dim); null; (|batch|,); (|input|,)
                blocks = [block.to(device) for block in blocks]
                # print(f'batch_num_inputs: {batch_num_inputs.shape}')
                train_batch_logits = model(blocks, lpa_labels, batch_num_inputs, batch_cat_inputs)
                mask = batch_labels == 2
                train_batch_logits = train_batch_logits[~mask]
                batch_labels = batch_labels[~mask]
                # batch_labels[mask] = 0

                train_loss = loss_fn(train_batch_logits, batch_labels)
                # backward
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.cpu().detach().numpy())

                if step % 10 == 0:
                    tr_batch_pred = torch.sum(torch.argmax(train_batch_logits.clone().detach(), dim=1) == batch_labels) / batch_labels.shape[0]
                    score = torch.softmax(train_batch_logits.clone().detach(), dim=1)[
                        :, 1].cpu().numpy()

                    # if (len(np.unique(score)) == 1):
                    #     print("all same prediction!")
                    try:
                        print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, '
                              'train_ap:{:.4f}, train_acc:{:.4f}, train_auc:{:.4f}'.format(epoch, step,
                                                                                           np.mean(
                                                                                               train_loss_list),
                                                                                           average_precision_score(
                                                                                               batch_labels.cpu().numpy(), score),
                                                                                           tr_batch_pred.detach(),
                                                                                           roc_auc_score(batch_labels.cpu().numpy(), score)))
                    except:
                        pass

            # mini-batch for validation
            val_loss_list = 0
            val_acc_list = 0
            val_all_list = 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_num_inputs, batch_cat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device)

                    blocks = [block.to(device) for block in blocks]
                    # print(f'batch_num_inputs (val): {batch_num_inputs.shape}')
                    val_batch_logits = model(blocks, lpa_labels, batch_num_inputs, batch_cat_inputs)
                    oof_predictions[seeds] = val_batch_logits
                    mask = batch_labels == 2
                    val_batch_logits = val_batch_logits[~mask]
                    batch_labels = batch_labels[~mask]
                    # batch_labels[mask] = 0
                    val_loss_list = val_loss_list + loss_fn(val_batch_logits, batch_labels)
                    # val_all_list += 1
                    val_batch_pred = torch.sum(torch.argmax(val_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                    val_acc_list = val_acc_list + val_batch_pred * torch.tensor(batch_labels.shape[0])
                    val_all_list = val_all_list + batch_labels.shape[0]
                    if step % 10 == 0:
                        score = torch.softmax(val_batch_logits.clone().detach(), dim=1)[
                            :, 1].cpu().numpy()
                        try:
                            print('In epoch:{:03d}|batch:{:04d}, val_loss:{:4f}, val_ap:{:.4f}, '
                                  'val_acc:{:.4f}, val_auc:{:.4f}'.format(epoch,
                                                                          step,
                                                                          val_loss_list/val_all_list,
                                                                          average_precision_score(
                                                                              batch_labels.cpu().numpy(), score),
                                                                          val_batch_pred.detach(),
                                                                          roc_auc_score(batch_labels.cpu().numpy(), score)))
                        except:
                            pass

            # val_acc_list/val_all_list, model)
            earlystoper.earlystop(val_loss_list/val_all_list, model)
            if earlystoper.is_earlystop:
                print("Early Stopping!")
                break
        print("Best val_loss is: {:.7f}".format(earlystoper.best_cv))
        test_ind = torch.from_numpy(np.array(test_idx)).long().to(device)
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dataloader = DataLoader(graph,
                                         test_ind,
                                         test_sampler,
                                         use_ddp=False,
                                         device=device,
                                         batch_size=args['batch_size'],
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=0,
                                         )
        b_model = earlystoper.best_model.to(device)
        b_model.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # print(input_nodes)
                batch_num_inputs, batch_cat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device)

                blocks = [block.to(device) for block in blocks]
                test_batch_logits = b_model(blocks, lpa_labels, batch_num_inputs, batch_cat_inputs)
                test_predictions[seeds] = test_batch_logits
                test_batch_pred = torch.sum(torch.argmax(test_batch_logits, dim=1) == batch_labels) / torch.tensor(batch_labels.shape[0])
                if step % 10 == 0:
                    print('In test batch:{:04d}'.format(step))
    mask = y_target == 2
    y_target[mask] = 0
    my_ap = average_precision_score(y_target, torch.softmax(
        oof_predictions, dim=1).cpu()[train_idx, 1])
    print("NN out of fold AP is:", my_ap)
    b_models, val_gnn_0, test_gnn_0 = earlystoper.best_model.to(
        'cpu'), oof_predictions, test_predictions

    test_score = torch.softmax(test_gnn_0, dim=1)[test_idx, 1].cpu().numpy()
    y_target = labels[test_idx].cpu().numpy()
    test_score1 = torch.argmax(test_gnn_0, dim=1)[test_idx].cpu().numpy()

    mask = y_target != 2
    test_score = test_score[mask]
    y_target = y_target[mask]
    test_score1 = test_score1[mask]

    print("test AUC:", roc_auc_score(y_target, test_score))
    print("test f1:", f1_score(y_target, test_score1, average="macro"))
    print("test AP:", average_precision_score(y_target, test_score))



def load_ggtan_data(dataset: str, test_size: float, device: str, gat_epochs=300, 
                    gat_learning_rate=0.01, gat_max_class_weight=4, gat_use_lpe=True):
    """
    Load graph, features, labels, and additional data for the given dataset.
    
    Args:
        dataset (str): Name of the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        device: The device to which tensors will be assigned.

    Returns:
        DataFrame: Feature data.
        Series: Labels.
        DGLGraph: Graph constructed from the data.
        list: List of categorical features.
        Tensor: Source node embeddings.
        Tensor: Target node embeddings.
    """
    # Define the path to the dataset
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")

    # Define the list of categorical features
    cat_features = ["Source", "Target", "Location", "Type"]

    # Load and preprocess the dataset
    df = pd.read_csv(prefix + f"{dataset}neofull.csv")
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    data = df[df["Labels"] <= 2].reset_index(drop=True)

    # Initialize lists for graph construction
    alls, allt = [], []
    edge_per_trans = 3  # Define the number of edges per transaction

    # Construct edges for the graph
    for column in ["Source", "Target", "Location", "Type"]:
        for c_id, c_df in data.groupby(column):
            c_df = c_df.sort_values(by="Time")
            sorted_idxs = c_df.index
            src = [sorted_idxs[i] for i in range(len(c_df)) 
                   for j in range(edge_per_trans) if i + j < len(c_df)]
            tgt = [sorted_idxs[i+j] for i in range(len(c_df))
                   for j in range(edge_per_trans) if i + j < len(c_df)]
            alls.extend(src)
            allt.extend(tgt)

    # Create the DGL graph
    g = dgl.graph((np.array(alls), np.array(allt)))

    # Encode categorical columns and prepare features and labels
    for col in ["Source", "Target", "Location", "Type"]:
        data[col] = LabelEncoder().fit_transform(data[col].apply(str).values)

    feat_data = data.drop("Labels", axis=1)
    labels = data["Labels"]

    # Save feature and label data
    feat_data.to_csv(prefix + f"{dataset}_feat_data.csv", index=False)
    labels.to_csv(prefix + f"{dataset}_label_data.csv", index=False)

    # Split data into training and test sets
    index = list(range(len(labels)))
    train_idx, test_idx, _, _ = train_test_split(index, labels, stratify=labels, test_size=test_size/2,
                                                 random_state=2, shuffle=True)

    # Define embedding dimensions and number of heads
    num_heads = 4
    emb_dim = (feat_data.shape[1] // 4) * 4  # Ensure emb_dim is a multiple of 4

    # Generate node embeddings
    source_emb, target_emb = generate_node_embedding(train_idx, test_idx, data, \
                                                     device, emb_dim, num_heads, 
                                                     epochs = gat_epochs,
                                                     learning_rate = gat_learning_rate,
                                                     max_class_weight = gat_max_class_weight,
                                                     gat_use_lpe=gat_use_lpe)

    return feat_data, labels, train_idx, test_idx, g, cat_features, source_emb, target_emb



def generate_node_embedding(train_idx, test_idx, data, device, emb_dim, num_heads, 
                            epochs=300, learning_rate=0.01, max_class_weight=4, gat_use_lpe=True):
    """
    Generate node embeddings using the EdgeGAT model.

    Args:
        train_idx: Indices for training data.
        test_idx: Indices for testing data.
        data: Input data for the model.
        device: The device to run the model on.
        emb_dim: Embedding dimension.
        num_heads: Number of heads in the GAT layer.
        epochs (int): Number of training epochs. Default is 300.
        learning_rate (float): Learning rate for the optimizer. Default is 0.01.
        max_class_weight (float): Maximum class weight for balancing. Default is 4.

    Returns:
        Tensor: Source node embeddings.
        Tensor: Target node embeddings.
    """
    # Define the path to save the model and embeddings
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "models/")

    # Load graph data and move to the specified device
    graph_data = load_gat_graph(data, device, gat_use_lpe).to(device)

    # Define the EdgeGAT model
    class EdgeGAT(torch.nn.Module):
        # Model initialization
        def __init__(self, num_node_features, num_edge_features, num_classes):
            super(EdgeGAT, self).__init__()
            # Define GAT layers
            self.conv1 = GATConv(num_node_features, emb_dim // num_heads, heads=num_heads, concat=True)
            self.conv2 = GATConv(emb_dim // num_heads * num_heads, emb_dim // num_heads, heads=num_heads, concat=True)
            # Define edge MLP for classification
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(num_edge_features + 2 * emb_dim // num_heads * num_heads, emb_dim // num_heads),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim // num_heads, num_classes)
            )

        # Forward pass of the model
        def forward(self, data, return_embedding=False):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            # First GAT layer
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            # Second GAT layer
            x = F.relu(self.conv2(x, edge_index))

            if return_embedding:
                return x  # Return node embeddings

            # Edge embeddings for classification
            edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1)
            return self.edge_mlp(edge_embeddings)

    # Initialize the model and move it to the specified device
    model = EdgeGAT(num_node_features=graph_data.num_node_features, 
                    num_edge_features=graph_data.edge_attr.size(1), 
                    num_classes=2).to(device)

    # Define the optimizer with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Calculate class weights for balancing and cap them at the specified maximum
    class_counts = [torch.sum(graph_data.y.cpu() == i).item() for i in range(2)]
    class_weights = [min(len(graph_data.y) / c, max_class_weight) for c in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define the loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        
        # Use only labeled edges for loss calculation
        train_mask = graph_data.y[train_idx] != 2
        loss = criterion(out[train_idx][train_mask], graph_data.y[train_idx][train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation at specified intervals
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_mask = graph_data.y[test_idx] != 2
                test_out = model(graph_data)
                test_loss = criterion(test_out[test_idx][test_mask], graph_data.y[test_idx][test_mask])

                # Calculate accuracy, precision, and AUC
                preds = test_out[test_idx][test_mask].max(1)[1]
                labels = graph_data.y[test_idx][test_mask]
                accuracy = accuracy_score(labels.cpu(), preds.cpu())
                auc = roc_auc_score(labels.cpu(), preds.cpu(), multi_class='ovo')
                f1 = f1_score(labels.cpu(), preds.cpu())

                # Print training and testing metrics
                print(f'Epoch {epoch}, Training Loss: {loss.item()}, Test Loss: {test_loss.item()}, '
                    f'Accuracy: {accuracy}, F1 Score: {f1}, AUC: {auc}')

    # Save the model state and embeddings
    torch.save(model.state_dict(), prefix + 'gat_model_state_dict.pth')

    # Generate and save node embeddings
    model.eval()
    with torch.no_grad():
        node_embeddings = model(graph_data, return_embedding=True)
        torch.save(node_embeddings, prefix + 'node_embeddings.pth')

    # Helper function to map edge embeddings
    def map_edge_embeddings(edge_index, node_embeddings):
        edge_index = edge_index.to(device)
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        return source_embeddings, target_embeddings

    # Load saved node embeddings and generate edge embeddings
    node_embeddings = torch.load(prefix + 'node_embeddings.pth').to(device)
    source_embeddings, target_embeddings = map_edge_embeddings(graph_data.edge_index, node_embeddings)

    return source_embeddings, target_embeddings



def load_gat_graph(data, device, gat_use_lpe=True):
    # Preparing edge features and labels
    edge_feat_data = data.drop(["Labels", "Source", "Target"], axis=1)

    # Merge Source and Target nodes
    num_source_nodes = data['Source'].nunique()
    data['Target'] += num_source_nodes  # Shift target indices

    # Preparing one-hot encoders for locations and types
    location_encoder = OneHotEncoder().fit(data[['Location']])
    type_encoder = OneHotEncoder().fit(data[['Type']])

    # Calculate features for Source and Target nodes separately
    source_features = calculate_node_features(data.groupby('Source'), location_encoder, type_encoder)
    target_features = calculate_node_features(data.groupby('Target'), location_encoder, type_encoder)

    # Combine Source and Target node features
    combined_node_features = np.vstack([source_features, target_features])
    node_features_tensor = torch.tensor(combined_node_features, dtype=torch.float)

    # Edge indices and features
    edge_src = torch.tensor(data['Source'].values, dtype=torch.long)
    edge_dst = torch.tensor(data['Target'].values, dtype=torch.long)
    edge_index = torch.stack([edge_src, edge_dst], dim=0).to(device)
    edge_features_tensor = torch.tensor(edge_feat_data.to_numpy(), dtype=torch.float)

    # Normalize node features
    scaler = StandardScaler()
    node_features_normalized = scaler.fit_transform(node_features_tensor.numpy())
    node_features_tensor = torch.tensor(node_features_normalized).float().to(device)

     # Normalize edge features
    scaler_edge = StandardScaler()
    edge_features_normalized = scaler_edge.fit_transform(edge_features_tensor.numpy())
    edge_features_tensor = torch.tensor(edge_features_normalized).float().to(device)

    # Labels
    labels_tensor = torch.tensor(data['Labels'].values, dtype=torch.long).to(device)

    # Create PyG Data object
    graph_data = Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_features_tensor, y=labels_tensor)

    # Create Laplacian Positional Encoding and Concat into Node Feature
    if gat_use_lpe:
        # Compute Laplacian Positional Encoding
        lpe = laplacian_positional_encoding(graph_data.edge_index, graph_data.num_nodes).to(device)

        # Concatenate LPE with node features
        node_features_with_lpe = torch.cat([graph_data.x, lpe], dim=1)

        # Normalize concatenated features
        scaler = StandardScaler()
        node_features_normalized = scaler.fit_transform(node_features_with_lpe.cpu().numpy())
        graph_data.x = torch.tensor(node_features_normalized).float().to(device)

    # # Instead of splitting node indices, split edge indices for train-test
    # edge_indices = np.arange(graph_data.edge_index.size(1))  # Number of edges

    return graph_data


def calculate_node_features(grouped_data, location_encoder, type_encoder):
    # Aggregations
    avg_amount = grouped_data['Amount'].mean()
    total_amount = grouped_data['Amount'].sum()
    std_amount = grouped_data['Amount'].std().fillna(0)
    num_transactions = grouped_data.size()
    num_locations = grouped_data['Location'].nunique()
    num_types = grouped_data['Type'].nunique()

    # Count the number of transactions for each location and type
    location_counts = grouped_data['Location'].value_counts().unstack(fill_value=0)
    type_counts = grouped_data['Type'].value_counts().unstack(fill_value=0)

    # One-hot encoding for locations and types
    one_hot_location = location_encoder.transform(location_counts.columns.values.reshape(-1, 1)).toarray()
    one_hot_type = type_encoder.transform(type_counts.columns.values.reshape(-1, 1)).toarray()

    # Multiply counts with one-hot encoding to create multi-hot encoding
    multi_hot_location = location_counts.values @ one_hot_location
    multi_hot_type = type_counts.values @ one_hot_type

    # Combine features
    features = pd.DataFrame({
        'avg_amount': avg_amount.values,
        'total_amount': total_amount.values,
        'std_amount': std_amount.values,
        'num_transactions': num_transactions.values,
        'num_locations': num_locations.values,
        'num_types': num_types.values
    }).reset_index(drop=True)

    # Append multi-hot encoded features
    features = pd.concat([features, pd.DataFrame(multi_hot_location), pd.DataFrame(multi_hot_type)], axis=1)
    features = features

    return features


def laplacian_positional_encoding_sparse(graph, dim):
    '''
    This is for the Transaction Graph (GTAN)
    '''
    print("Starting Laplacian Positional Encoding...")

    # Get the adjacency matrix of the graph in Scipy CSR format
    adj_matrix_sparse = graph.adj_external(scipy_fmt='csr')

    # Calculate the degree matrix
    degrees = adj_matrix_sparse.sum(axis=1).A1  # Convert to 1D array
    D_sparse = sp.diags(degrees)

    # Compute the normalized Laplacian (L = D - A) in sparse format
    L_sparse = D_sparse - adj_matrix_sparse

    print("Starting eigendecomposition...")
    start_time = time.time()

    # Compute the first 'dim' eigenvectors using sparse eigendecomposition
    eigenvalues, eigenvectors = eigsh(L_sparse, k=dim, which='SM')

    end_time = time.time()
    print(f"Eigendecomposition completed in {end_time - start_time:.2f} seconds.")

    return torch.from_numpy(eigenvectors).float()


def random_walk_structural_encoding(graph, walk_length, dim):
    '''
    This is for the Transaction Graph (GTAN)
    '''
    num_nodes = graph.num_nodes()
    context = torch.zeros((num_nodes, dim))

    # Perform random walks
    for node in tqdm(range(num_nodes), desc="Computing RWSE"):
        walks = dgl.sampling.random_walk(graph, [node] * walk_length, length=walk_length)[0]
        
        # Count visits
        for walk in walks:
            for visit in walk:
                if visit >= 0:  # Valid node (not a padded node in the walk)
                    context[node, visit % dim] += 1

    # Normalize by walk length
    context /= walk_length
    return context


def laplacian_positional_encoding(edge_index, num_nodes, k=20):
    '''
    This is for the Source-Target Graph (GAT)
    '''
    # Convert edge_index to a scipy sparse matrix
    adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    # Compute Laplacian
    laplacian = scipy.sparse.csgraph.laplacian(adj_matrix, normed=True)
    # Compute the first k eigenvectors
    eigenvalues, eigenvectors = eigs(laplacian, k=k, which='SM')
    return torch.from_numpy(np.real(eigenvectors)).float()

