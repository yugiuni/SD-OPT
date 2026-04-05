import numpy as np
import argparse
import torch
import random
import time
from model import CLIP
from torch import nn, optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from multitask import multitask_data_generator
from model_g_coop import CoOp
from data_graph import DataHelper
from torch.utils.data import DataLoader
import graph_Orthoprompt as Prompt
import torch.nn.functional as F
import model
from simple_tokenizer import SimpleTokenizer as _Tokenizer


def center_norm(datas, center=True):
    if center:
        datas = datas - datas.mean(0, keepdim=True)
        datas = datas / torch.norm(datas, dim=1, keepdim=True)
    return datas


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args, node_feat, node_f1, text_feat, text_feat1):
    setup_seed(args.seed)

    clip_model = CLIP(args)
    clip_model.load_state_dict(torch.load(f'./res/{data_name}/node_ttgt_8&12_0.1.pkl', map_location=device))

    task_list, train_idx, val_idx, test_idx = multitask_data_generator(lab_list, labeled_ids, labels, args.k_spt,
                                                                       args.k_val, args.k_qry, args.n_way)

    all_acc = []
    f1_list = []
    for j in range(len(task_list)):
        train_idx_ts = torch.from_numpy(np.array(train_idx[j])).to(device)
        val_idx_ts = torch.from_numpy(np.array(val_idx[j])).to(device)
        test_idx_ts = torch.from_numpy(np.array(test_idx[j])).to(device)

        train_truth = np.array(lab_list)[np.array(train_idx[j])]
        val_truth = np.array(lab_list)[np.array(val_idx[j])]
        test_truth = np.array(lab_list)[np.array(test_idx[j])]

        task_lables_arr = np.array(labels)[task_list[j]]
        task_labels_dict = {task_lables_arr[i]: i for i in range(task_lables_arr.shape[0])}

        train_truth_ts = torch.from_numpy(np.array([task_labels_dict[train_truth[i]] for i in range(len(train_truth))])).to(device)
        val_truth_ts = torch.from_numpy(np.array([task_labels_dict[val_truth[i]] for i in range(len(val_truth))])).to(device)
        test_truth_ts = torch.from_numpy(np.array([task_labels_dict[test_truth[i]] for i in range(len(test_truth))])).to(device)

        task_lables = task_lables_arr.tolist()
        Data = DataHelper(arr_edge_index, args, train_idx[j])
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for sample_batched in loader:
            s_n = sample_batched['s_n'].numpy()
            t_n = sample_batched['t_n'].numpy()
        s_n = s_n.reshape(args.num_labels, args.k_spt)
        t_n = t_n.reshape(args.num_labels, args.k_spt * args.neigh_num)
        temp = [np.concatenate((s_n[i], t_n[i])) for i in range(args.num_labels)]

        g_texts = []
        for i in range(len(temp)):
            g_text = [tit_list[a] for a in temp[i]]
            g_texts.append(g_text)

        prompt = Prompt.GPFplusAtt(args.embed_dim, args.pnum, args.T1, args.group1, args.k).to(device)
        model = CoOp(args, task_lables, clip_model, g_texts, device)

        model_param_group = [{"params": prompt.parameters()}]
        optimizer_prompt = optim.Adam(model_param_group, lr=args.graphprompt_lr, weight_decay=args.weight_decay, amsgrad=False)

        best_val = 0
        patience = 10
        counter = 0
        model_path = f'./res/{data_name}/g_coop.pkl'

        for epoch in range(1, args.ft_epoch + 1):
            model.train()
            optimizer_prompt.zero_grad()
            model.forward(args.Ortho, train_idx_ts, prompt.cat(node_feat, node_f1, text_feat, text_feat1, args.Ortho),
                          edge_index, train_truth_ts, args.lamda)
            optimizer_prompt.step()

            model.eval()
            with torch.no_grad():
                res = model.forward(args.Ortho, val_idx_ts,
                                    prompt.cat(node_feat, node_f1, text_feat, text_feat1, args.Ortho, training=False),
                                    edge_index, val_truth_ts, args.lamda, training=False)
                val_acc = accuracy_score(val_truth_ts.cpu(), res.argmax(dim=1).cpu())
                if val_acc <= best_val:
                    counter += 1
                    if counter >= patience:
                        break
                else:
                    best_val = val_acc
                    torch.save(model, model_path)
                    counter = 0

        best_model = torch.load(model_path)
        best_model.eval()
        with torch.no_grad():
            res = model.forward(args.Ortho, test_idx_ts,
                                prompt.cat(node_feat, node_f1, text_feat, text_feat1, args.Ortho, training=False),
                                edge_index, test_truth_ts, args.lamda, training=False)
            test_acc = accuracy_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu())
            all_acc.append(test_acc)
            f1_list.append(f1_score(test_truth_ts.cpu(), res.argmax(dim=1).cpu(), average='macro'))

    print('acc:', round(np.mean(all_acc).item(), 4))
    print('macro f1:', round(np.mean(f1_list).item(), 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregation_times', type=int, default=2)
    parser.add_argument('--ft_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gnn_input', type=int, default=128)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--edge_coef', type=float, default=0.1)
    parser.add_argument('--neigh_num', type=int, default=3)
    parser.add_argument('--num_labels', type=int, default=5)
    parser.add_argument('--k_spt', type=int, default=5)
    parser.add_argument('--k_val', type=int, default=5)
    parser.add_argument('--k_qry', type=int, default=50)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--coop_n_ctx', type=int, default=4)
    parser.add_argument('--prompt_lr', type=float, default=0.01)
    parser.add_argument('--position', type=str, default='end')
    parser.add_argument('--class_specific', type=bool, default=False)
    parser.add_argument('--ctx_init', type=bool, default=True)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=49408)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pnum', type=int, default=20)
    parser.add_argument('--graphprompt_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lamda', type=float, default=0)
    parser.add_argument('--T1', type=int, default=4)
    parser.add_argument('--group1', type=int, default=1)
    parser.add_argument('--Ortho', default=True)
    parser.add_argument('--k', type=float, default=1.0)

    args = parser.parse_args()

    data_name = 'cora'
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    num_nodes = 0
    tit_list = []
    lab_list = []
    with open('./data/train_text.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            tit_list.append(parts[2])
            lab_list.append(parts[3])
            num_nodes += 1

    print('num_nodes', num_nodes)

    labeled_ids = [i for i, lab in enumerate(lab_list) if lab != 'nan']
    print('{} nodes having lables'.format(len(labeled_ids)))

    node_f = np.load('./data/node_f.npy')
    node_f = preprocessing.StandardScaler().fit_transform(node_f)
    node_f = torch.from_numpy(node_f).to(device)
    node_f1 = center_norm(node_f)

    text_f = np.load('./data/text_feature/cora_test_feature.npy')
    text_f = preprocessing.StandardScaler().fit_transform(text_f)
    text_f = torch.from_numpy(text_f).to(device)
    text_f1 = center_norm(text_f)

    raw_edge_index = [[], []]
    with open('./data/mapped_edges.txt', 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            raw_edge_index[0].append(u)
            raw_edge_index[1].append(v)
    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    arr_edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(np.array(edge_index)).to(device)

    with open('./data/lab_list.txt', 'r') as f:
        label_texts = f.readline().strip().split('\t')

    labels = [lab for lab in label_texts if lab != 'nan']

    start = time.perf_counter()
    main(args, node_f, node_f1, text_f, text_f1)
    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))