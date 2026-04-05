import random
import numpy as np

def multitask_data_generator(labels, labeled_node_list, select_array, k_spt, k_val, k_qry, n_way):
    class_idx_list = [[] for _ in range(len(select_array))]
    train_class_list = [[] for _ in range(len(select_array))]
    val_class_list = [[] for _ in range(len(select_array))]
    test_class_list = [[] for _ in range(len(select_array))]

    for j, node in enumerate(labeled_node_list):
        for i, cls in enumerate(select_array):
            if labels[j] == cls:
                class_idx_list[i].append(node)
                break

    usable_labels = [i for i, idx_list in enumerate(class_idx_list) if len(idx_list) >= 30]

    random.shuffle(usable_labels)
    task_list = [usable_labels[i * n_way:(i + 1) * n_way] for i in range(len(usable_labels) // n_way)]

    usable_set = set(usable_labels)
    for i in range(len(select_array)):
        if i not in usable_set:
            continue
        train_class_list[i] = np.random.choice(class_idx_list[i], k_spt, replace=False).tolist()
        val_candidates = [n for n in class_idx_list[i] if n not in train_class_list[i]]
        val_class_list[i] = np.random.choice(val_candidates, k_val, replace=False).tolist()
        test_candidates = [n for n in class_idx_list[i] if n not in train_class_list[i] and n not in val_class_list[i]]
        test_class_list[i] = test_candidates

    train_idx = []
    val_idx = []
    test_idx = []
    for task in task_list:
        train_idx.append(sum((train_class_list[j] for j in task), []))
        val_idx.append(sum((val_class_list[j] for j in task), []))
        test_idx.append(sum((test_class_list[j] for j in task), []))

    return task_list, train_idx, val_idx, test_idx