import copy


def load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device):
    batch_num_inputs = num_feat[input_nodes].to(device)
    batch_cat_inputs = {i: cat_feat[i][input_nodes].to(device) for i in cat_feat if i not in {"Labels"}}
    # for i in batch_cat_inputs:
    #    print(batch_cat_inputs[i].shape)
    batch_labels = labels[seeds].to(device)
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]
    propagate_labels[:seeds.shape[0]] = 2
    return batch_num_inputs, batch_cat_inputs, batch_labels, propagate_labels.to(device)
