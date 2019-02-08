import torch
import torch.utils.data
import datasets

def make_datasets(train_data, val_data, test_data):
    train_tensor = torch.from_numpy(train_data)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(val_data)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(test_data)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)

    return train_dataset, valid_dataset, test_dataset

def make_loaders(train_dataset, valid_dataset, test_dataset, batch_size, test_batch_size, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs)
    return train_loader, valid_loader, test_loader

def make_inverse_datasets(model, train_dataset, valid_dataset, test_dataset):
    train_X = train_dataset.tensors
    train_Z = model(train_X)
    train_dataset_inverse = torch.utils.data.TensorDataset(train_X, train_Z)

    valid_X = valid_dataset.tensors
    valid_Z = model(valid_X)
    valid_dataset_inverse = torch.utils.data.TensorDataset(valid_X, valid_Z)

    test_X = test_dataset.tensors
    test_Z = model(test_X)
    test_dataset_inverse = torch.utils.data.TensorDataset(test_X, test_Z)

    return train_dataset_inverse, valid_dataset_inverse, test_dataset_inverse



def import_dataset(batch_size, test_batch_size, random_seed, cuda=False):
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed(random_seed)
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    dataset = getattr(datasets, 'POWER')()

    train_dataset, valid_dataset, test_dataset = make_datasets(dataset.trn.x, dataset.val.x, dataset.tst.x)
    train_loader, valid_loader, test_loader = make_loaders(train_dataset, valid_dataset, test_dataset,
                                                           batch_size, test_batch_size, **kwargs)

    return train_loader, valid_loader, test_loader