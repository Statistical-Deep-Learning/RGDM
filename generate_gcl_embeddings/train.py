# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import random
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from utils import process
from utils import aug
from modules.gcn import GCNLayer
from net.merit import MERIT
import json
import torch.nn as nn
import torch.optim as optim

def load_npz(file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--data', type=str, default='citeseer')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=10)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=2000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--sparse', type=str_to_bool, default=True)

parser.add_argument('--input_dim', type=int, default=3703)
parser.add_argument('--gnn_dim', type=int, default=64)
parser.add_argument('--proj_dim', type=int, default=512)
parser.add_argument('--proj_hid', type=int, default=4096)
parser.add_argument('--pred_dim', type=int, default=512)
parser.add_argument('--pred_hid', type=int, default=4096)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--drop_edge', type=float, default=0.4)
parser.add_argument('--drop_feat1', type=float, default=0.4)
parser.add_argument('--drop_feat2', type=float, default=0.4)

parser.add_argument('--embed_path', type=str, default='./')
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--split_path', type=str, default='./')
parser.add_argument('--att_adj_path', type=str, default='./')
parser.add_argument('--att_type', type=str, default='./')
parser.add_argument('--att_rate', type=float, default=0.25)


args = parser.parse_args()
torch.set_num_threads(4)

class classifier(nn.Module):
    def __init__(self, ft_size, num_classes):
        super(classifier, self).__init__()
        self.fc = nn.Linear(ft_size, num_classes, bias=False)
    def forward(self, x):
        x = self.fc(x)
        return x

def evaluation(labels, adj, diff, feat, gnn, idx_train, idx_test, sparse, best_acc_all=0.0):
    model = GCNLayer(input_size, gnn_output_size)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)
        embeds2 = model(feat, diff, sparse)

    learning_rate = 0.01
    weight_decay = 5e-6
    epoch_num = 5000
    all_embeds = embeds1[0,:] + embeds2[0,:]
    feat_dim = all_embeds.shape[1]
    num_classes = np.max(labels) + 1
    model = classifier(feat_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    feat_torch = torch.FloatTensor(all_embeds).to(device)
    labels_torch = torch.LongTensor(labels).to(device)
    best_acc = 0.0
    pseudo_labels = np.zeros(labels.shape[0])
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        model.train()
        out = model(feat_torch)
        loss = criterion(out[idx_train,:], labels_torch[idx_train])
        loss.backward()
        optimizer.step()   
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                out = model(feat_torch)
                test_loss = criterion(out[idx_test,:], labels_torch[idx_test]) 
                predict_lables = torch.argmax(out, dim=-1)
                correct = predict_lables[idx_test].eq(labels_torch[idx_test]).sum().item()
                acc = correct / predict_lables[idx_test].shape[0]
                # if epoch % 10 == 0:
                #     print(f'Epoch[{epoch}] Loss: {loss.item():.8}  Test_Loss: {test_loss.item():.8} ACC: {acc:.8}')
                if acc > best_acc:
                    best_acc = acc
                    pseudo_labels = torch.argmax(out, dim=-1).cpu().numpy()
    pseudo_labels[idx_train] = labels[idx_train]
    if best_acc > best_acc_all:
        print('Saving embeddings for model with acc {:.5f}...'.format(acc))
        np.save( args.embed_path + args.att_type +'_'+ str(args.att_rate) + '_'+ 'all_embs.npy', all_embeds)
        np.save( args.embed_path + args.att_type +'_'+ str(args.att_rate) + '_'+ 'all_ps_labels.npy', pseudo_labels)
    return best_acc


if __name__ == '__main__':

    if not os.path.exists(args.embed_path):
        os.makedirs(args.embed_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_runs = args.runs
    eval_every_epoch = args.eval_every

    dataset = args.data
    input_size = args.input_dim

    gnn_output_size = args.gnn_dim
    projection_size = args.proj_dim
    projection_hidden_size = args.proj_hid
    prediction_size = args.pred_dim
    prediction_hidden_size = args.pred_hid
    momentum = args.momentum
    beta = args.beta
    alpha = args.alpha

    drop_edge_rate_1 = args.drop_edge
    drop_feature_rate_1 = args.drop_feat1
    drop_feature_rate_2 = args.drop_feat2

    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    batch_size = args.batch_size
    patience = args.patience

    sparse = args.sparse

    # Loading dataset
    # adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    #read unattacked data
    _, features, labels = load_npz(args.data_path)
    #read attacked adj from args.att_adj_path
    adj = sp.load_npz(args.att_adj_path)

    # Open the file and load the data
    with open(args.split_path, 'r') as file:
        data_split = json.load(file)

    idx_train = data_split['idx_train']
    idx_val = data_split['idx_val']
    idx_test = data_split['idx_test']
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # if os.path.exists('data/diff_{}_{}.npy'.format(dataset, alpha)):
    #     diff = np.load('data/diff_{}_{}.npy'.format(dataset, alpha), allow_pickle=True)
    # else:
    #     diff = aug.gdc(adj, alpha=alpha, eps=0.0001)
    #     np.save('data/diff_{}_{}'.format(dataset, alpha), diff)
    diff = aug.gdc(adj, alpha=alpha, eps=0.0001)

    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = np.max(labels) + 1

    features = torch.FloatTensor(features[np.newaxis])
    # labels = torch.FloatTensor(labels[np.newaxis])

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)
        eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    result_over_runs = []
    
    # Initiate models
    model = GCNLayer(input_size, gnn_output_size)
    merit = MERIT(gnn=model,
                  feat_size=input_size,
                  projection_size=projection_size,
                  projection_hidden_size=projection_hidden_size,
                  prediction_size=prediction_size,
                  prediction_hidden_size=prediction_hidden_size,
                  moving_average_decay=momentum, beta=beta).to(device)

    opt = torch.optim.Adam(merit.parameters(), lr=lr, weight_decay=weight_decay)

    results = []

    # Training
    best = 0
    patience_count = 0
    for epoch in range(epochs):
        for _ in range(batch_size):
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1)
            ba = adj[idx: idx + sample_size, idx: idx + sample_size]
            bd = diff[idx: idx + sample_size, idx: idx + sample_size]
            bd = sp.csr_matrix(np.matrix(bd))
            features = features.squeeze(0)
            bf = features[idx: idx + sample_size]

            aug_adj1 = aug.aug_random_edge(ba, drop_percent=drop_edge_rate_1)
            aug_adj2 = bd
            aug_features1 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_1)
            aug_features2 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_2)

            aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
            aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

            if sparse:
                adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
                adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
            else:
                aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
                aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()
                adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
                adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)

            aug_features1 = aug_features1.to(device)
            aug_features2 = aug_features2.to(device)

            opt.zero_grad()
            loss = merit(adj_1, adj_2, aug_features1, aug_features2, sparse)
            loss.backward()
            opt.step()
            merit.update_ma()

        if epoch % eval_every_epoch == 0:
            acc = evaluation(labels, eval_adj, eval_diff, features, model, idx_train, idx_test, sparse, best)
            if acc > best:
                best = acc
                patience_count = 0
            else:
                patience_count += 1
            results.append(acc)
            print('\t epoch {:03d} | loss {:.5f} | clf test acc {:.5f}'.format(epoch, loss.item(), acc))
            if patience_count >= patience:
                print('Early Stopping.')
                break
            
    result_over_runs.append(max(results))
    print('\t best acc {:.5f}'.format(max(results)))