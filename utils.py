import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
import time
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize

EOS = 1e-10


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask, samples


def normalize_adj(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def get_adj_from_edges(edges, weights, nnodes):
    adj = torch.zeros(nnodes, nnodes).cuda()
    adj[edges[0], edges[1]] = weights
    return adj


def augmentation(features_1, adj_1, features_2, adj_2, args, training):
    # view 1
    mask_1, _ = get_feat_mask(features_1, args.maskfeat_rate_1)
    features_1 = features_1 * (1 - mask_1)
    if not args.sparse:
        adj_1 = F.dropout(adj_1, p=args.dropedge_rate_1, training=training)
    else:
        adj_1.edata['w'] = F.dropout(adj_1.edata['w'], p=args.dropedge_rate_1, training=training)

    # # view 2
    mask_2, _ = get_feat_mask(features_1, args.maskfeat_rate_2)
    features_2 = features_2 * (1 - mask_2)
    if not args.sparse:
        adj_2 = F.dropout(adj_2, p=args.dropedge_rate_2, training=training)
    else:
        adj_2.edata['w'] = F.dropout(adj_2.edata['w'], p=args.dropedge_rate_2, training=training)

    return features_1, adj_1, features_2, adj_2


def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges.cuda()


def eval_debug_mode(embedding, labels, train_mask, val_mask, test_mask):

    t1 = time.time()

    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()

    X = normalize(X, norm='l2')

    nb_split = train_mask.shape[1]

    accs = []
    for split in range(nb_split):
        X_train = X[train_mask.cpu()[:, split]]
        X_test = X[test_mask.cpu()[:, split]]
        y_train = Y[train_mask.cpu()[:, split]]
        y_test = Y[test_mask.cpu()[:, split]]

        logreg = LogisticRegression(solver='liblinear')
        c = 2.0 ** np.arange(-10, 10)
        clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                           param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                           verbose=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)

    print('eval time:{:.4f}s'.format(time.time() - t1))

    return accs


def eval_test_mode(embedding, labels, train_mask, val_mask, test_mask):

    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    X_train = X[train_mask.cpu()]
    X_val = X[val_mask.cpu()]
    X_test = X[test_mask.cpu()]
    y_train = Y[train_mask.cpu()]
    y_val = Y[val_mask.cpu()]
    y_test = Y[test_mask.cpu()]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    return acc_test * 100, acc_val * 100

