from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from data.dataset import ISICDataset
from data.fssl_liver_dataset import LiverDataset
# from data.fixmatch_liver_dataset import LiverDataset

def get_dataloader(dataset_type, data_path1, data_path2, image_size, batch_size, data_parties, labeled_ratio):
    if dataset_type == "json":
        # json 方式读取
        return get_data_by_json(data_path1, image_size, batch_size, data_parties, labeled_ratio)
    elif dataset_type == "random":
        return get_data_by_random(data_path1, data_path2, image_size, batch_size, data_parties, labeled_ratio)
    elif dataset_type == "ssl":
        return get_all_train_and_test_data(data_path1, image_size, batch_size, data_parties)
    else:
        assert "参数错误"

def sequential_split_list(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of lengths must be equal to the length of the dataset")

    indices = list(range(len(dataset)))
    split_indices = []
    start_idx = 0

    for length in lengths:
        split_indices.append(indices[start_idx:start_idx + length])
        start_idx += length

    subsets = [Subset(dataset, indices) for indices in split_indices]
    return subsets

def sequential_split(dataset, split_ratio=0.3):
    dataset_size = len(dataset)
    split_index = int(dataset_size * split_ratio)
    indices = list(range(dataset_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset

def split_dataset(train_set, subset_num):
    # 定义要拆分的长度
    num_splits = subset_num
    subset_length = len(train_set) // num_splits
    remainder = len(train_set) % num_splits  # 计算余数
    # 使用 random_split 函数拆分数据集
    lengths_list = [subset_length] * (num_splits - 1) + [subset_length + remainder]  # 将余数添加到最后一个子数据集
    train_sub_datasets = sequential_split_list(train_set, lengths_list)

    # 将子数据集形成一个列表
    sub_datasets_list = list(train_sub_datasets)
    return sub_datasets_list

# 方式一：读取json方式获取数据集
def get_data_by_json(data_path, image_size, batch_size, data_parties, labeled_ratio):
    test_dl = get_test_data(data_path, image_size, batch_size, data_parties)
    local_dls = get_data_loader_by_json(data_path, image_size, batch_size, data_parties, labeled_ratio)

    return local_dls, test_dl

def get_train_data(data_path, image_size, batch_size, client_id, labeled_ratio):
    labeled_train_set = LiverDataset(image_path=data_path, stage='train', image_size=image_size,
                                     is_augmentation=True, type="labeled_train", client_id=client_id, label_ratio=labeled_ratio)
    unlabeled_train_set = LiverDataset(image_path=data_path, stage='train', image_size=image_size,
                                     is_augmentation=True, type="unlabeled_train", client_id=client_id, label_ratio=labeled_ratio)
    train_set = labeled_train_set+unlabeled_train_set
    # todo 需要注释
    # if labeled_ratio != 0.3:
    #     labeled_train_set, _ = random_split(labeled_train_set, [int(len(labeled_train_set) * labeled_ratio),
    #                                                        len(labeled_train_set) - int(len(labeled_train_set) * labeled_ratio)])
    print('before:', len(labeled_train_set), len(train_set))
    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    assert len(labeled_train_set) == len(train_set)
    print('after:', len(labeled_train_set), len(train_set))
    # todo 需要修改
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=1, batch_size=batch_size,  shuffle=True, pin_memory=True, drop_last=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return train_labeled_dataloder, train_unlabeled_dataloder

def get_data_loader_by_json(data_path, image_size, batch_size, data_parties, labeled_ratio):
    # 创建用于存储 DataLoader 的列表
    data_loader_list = []
    for client_id in range(1, data_parties+1):
        test_set = LiverDataset(image_path=data_path, stage='test', image_size=image_size, is_augmentation=False,
                                            type="test", client_id=client_id)

        test_loder = DataLoader(dataset=test_set, num_workers=1, batch_size=batch_size, shuffle=False, pin_memory=True)

        train_labeled_loder, train_unlabeled_loder = get_train_data(data_path, image_size, batch_size, client_id, labeled_ratio)

        data_loader_list.append((train_labeled_loder, train_unlabeled_loder, test_loder))

    '''
    # 拆分数据集
    train_set_list = split_dataset(train_set, subset_num)
    test_set_list = split_dataset(test_set, subset_num)
    # 创建用于存储 DataLoader 的列表
    data_loader_list = []

    # 遍历每个子数据集
    for idx, sub_dataset in enumerate(train_set_list):
        labeled_loader, unlabeled_loader = get_train_sub_data(sub_dataset, batch_size, labeled_ratio)

        test_sub_dataset = test_set_list[idx]
        test_loder = DataLoader(dataset=test_sub_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

        data_loader_list.append((labeled_loader, unlabeled_loader, test_loder))
    '''

    return data_loader_list

def get_test_data(data_path, image_size, batch_size, data_parties):
    data_set_list = []
    for client_id in range(1, data_parties + 1):
        test_set = LiverDataset(image_path=data_path, stage='test', image_size=image_size, is_augmentation=False,
                                type="test", client_id=client_id)
        data_set_list.append(test_set)

    combined_test_set = ConcatDataset(data_set_list)
    test_loder = DataLoader(dataset=combined_test_set, num_workers=1, batch_size=batch_size, shuffle=False, pin_memory=True)
    return test_loder

# 方式二：按照随机目录的方式划分数据集
def get_data_by_random(data_path1, data_path2, image_size, batch_size, data_parties, labeled_ratio):
    local_dls1 = get_data_loader(data_path1, image_size, batch_size, data_parties, labeled_ratio)
    local_dls2 = get_data_loader(data_path2, image_size, batch_size, data_parties, labeled_ratio)

    test_dl = get_test_data_loader(data_path1, data_path2, image_size, batch_size)

    return local_dls1+local_dls2, test_dl

def get_train_sub_data(train_set, batch_size, labeled_percentage):
    # train_set = ISICDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True)
    # labeled_train_set, unlabeled_train_set = random_split(train_set, [int(len(train_set) * labeled_percentage),
    #                                                    len(train_set) - int(len(train_set) * labeled_percentage)])
    labeled_train_set, unlabeled_train_set = random_split(train_set,[int(len(train_set) * labeled_percentage),
                                                                              len(train_set) - int(len(train_set) * labeled_percentage)])
    print('before:', len(labeled_train_set), len(train_set))
    # todo 这里后续要改回去
    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    # labeled_ratio = len(train_set) // len(labeled_train_set)
    # labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    # labeled_train_set = ConcatDataset([labeled_train_set,
    #                                    Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    # assert len(labeled_train_set) == len(train_set)
    print('after:', len(labeled_train_set), len(train_set))
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=1, batch_size=batch_size,  shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_labeled_dataloder, train_unlabeled_dataloder


def get_data_loader(data_path, image_size, batch_size, data_parties, labeled_ratio=0.3):
    # 拆分数据集
    train_set = ISICDataset(image_path=data_path, stage='train', image_size=image_size, is_augmentation=True)
    test_set = ISICDataset(image_path=data_path, stage='test', image_size=image_size, is_augmentation=False)
    train_set_list = split_dataset(train_set, data_parties)
    test_set_list = split_dataset(test_set, data_parties)
    # 创建用于存储 DataLoader 的列表
    data_loader_list = []
    # 遍历每个子数据集
    for idx, sub_dataset in enumerate(train_set_list):
        labeled_loader, unlabeled_loader = get_train_sub_data(sub_dataset, batch_size, labeled_ratio)

        test_sub_dataset = test_set_list[idx]
        test_loder = DataLoader(dataset=test_sub_dataset, num_workers=1, batch_size=batch_size, shuffle=False, pin_memory=True)

        data_loader_list.append((labeled_loader, unlabeled_loader, test_loder))

    return data_loader_list

def get_test_data_loader(data_path1, data_path2, image_size, batch_size):
    # 拆分数据集
    test_set1 = ISICDataset(image_path=data_path1, stage='test', image_size=image_size, is_augmentation=False)
    test_set2 = ISICDataset(image_path=data_path2, stage='test', image_size=image_size, is_augmentation=False)
    test_set = ConcatDataset([test_set1, test_set2])
    test_loder = DataLoader(dataset=test_set, num_workers=1, batch_size=batch_size, shuffle=False, pin_memory=True)

    return test_loder

# 方式三：读取ssl数据，对每个客户端的数据合并
def get_all_train_and_test_data(data_path, image_size, batch_size, data_parties):
    test_data_set_list = []
    label_train_data_set_list = []
    unlabel_train_data_set_list = []
    for client_id in range(1, data_parties + 1):
        test_set = LiverDataset(image_path=data_path, stage='test', image_size=image_size, is_augmentation=False,
                                type="test", client_id=client_id)
        test_data_set_list.append(test_set)

        labeled_train_set = LiverDataset(image_path=data_path, stage='train', image_size=image_size,
                                         is_augmentation=True, type="labeled_train", client_id=client_id)
        unlabeled_train_set = LiverDataset(image_path=data_path, stage='train', image_size=image_size,
                                           is_augmentation=True, type="unlabeled_train", client_id=client_id)
        # train_set = labeled_train_set + unlabeled_train_set
        label_train_data_set_list.append(labeled_train_set)
        unlabel_train_data_set_list.append(unlabeled_train_set)

    combined_test_set = ConcatDataset(test_data_set_list)
    combined_label_train_set = ConcatDataset(label_train_data_set_list)
    combined_unlabel_train_set = ConcatDataset(unlabel_train_data_set_list)

    # todo 这里后续注释掉
    # 扩充数据集
    print('before:', len(combined_label_train_set), len(combined_unlabel_train_set))
    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    labeled_ratio = len(combined_unlabel_train_set) // len(combined_label_train_set)
    combined_label_train_set = ConcatDataset([combined_label_train_set for i in range(labeled_ratio)])
    combined_label_train_set = ConcatDataset([combined_label_train_set,
                                       Subset(combined_label_train_set, range(len(combined_unlabel_train_set) - len(combined_label_train_set)))])
    assert len(combined_label_train_set) == len(combined_unlabel_train_set)
    print('after:', len(combined_label_train_set), len(combined_unlabel_train_set))

    test_loder = DataLoader(dataset=combined_test_set, num_workers=1, batch_size=batch_size, shuffle=False, pin_memory=True)
    label_train_loder = DataLoader(dataset=combined_label_train_set, num_workers=1, batch_size=batch_size, shuffle=True,
                                           pin_memory=True, drop_last=True)
    unlabel_train_loder = DataLoader(dataset=combined_unlabel_train_set, num_workers=1, batch_size=batch_size, shuffle=True,
                                   pin_memory=True, drop_last=True)
    return label_train_loder, unlabel_train_loder, test_loder

