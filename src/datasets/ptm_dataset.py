import json
import h5py
import random
import torch.utils.data as data


def hdf5_to_list(hdf5_filepath):
    data_list = []
    with h5py.File(hdf5_filepath, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            group = hdf5_file[key]
            item_dict = {}
            for sub_key in group.keys():
                if sub_key == 'coords_chain_A':
                    coords_chain_A_dict = {}
                    for coords_key in group[sub_key].keys():
                        coords_chain_A_dict[coords_key] = group[sub_key][coords_key][()].tolist()
                    item_dict[sub_key] = coords_chain_A_dict
                elif sub_key == 'id' or sub_key == 'seq':
                    item_dict[sub_key] = group[sub_key][()].astype(str)[0]
                else:
                    item_dict[sub_key] = group[sub_key][()].tolist()
            data_list.append(item_dict)
    return data_list


class PTMDataset(data.Dataset):
    def __init__(self, path='./',  split='train', max_length=500, test_name='All', data = None):
        self.path = path
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        datapath=""
        if split == "train":
            datapath = path + "/train.hdf5"
        elif split == "test":
            datapath = path + "/test.hdf5"
        elif split == "valid":
            datapath = path + "/val.hdf5"
        elif split == "predict":
            datapath = path + "/predict.hdf5"

        self.data = hdf5_to_list(datapath)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if L>self.max_length:
            if self.mode=="predict":
                item['seq'] = item['seq'][:self.max_length]
                item['coords_chain_A']["CA_chain_A"] = item['coords_chain_A']["CA_chain_A"][:self.max_length]
                item['coords_chain_A']["C_chain_A"] = item['coords_chain_A']["C_chain_A"][:self.max_length]
                item['coords_chain_A']["O_chain_A"] = item['coords_chain_A']["O_chain_A"][:self.max_length]
                item['coords_chain_A']["N_chain_A"] = item['coords_chain_A']["N_chain_A"][:self.max_length]
                item['chain_mask'] = ([1]*len(self.data))[:self.max_length]
                item['chain_encoding'] = ([1]*len(self.data))[:self.max_length]
                item["ptm"]=item["ptm"][:self.max_length] 
            else:
                max_index = L - self.max_length
                truncate_index = random.randint(0, max_index)
                item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
                item['coords_chain_A']["CA_chain_A"] = item['coords_chain_A']["CA_chain_A"][truncate_index:truncate_index+self.max_length]
                item['coords_chain_A']["C_chain_A"] = item['coords_chain_A']["C_chain_A"][truncate_index:truncate_index+self.max_length]
                item['coords_chain_A']["O_chain_A"] = item['coords_chain_A']["O_chain_A"][truncate_index:truncate_index+self.max_length]
                item['coords_chain_A']["N_chain_A"] = item['coords_chain_A']["N_chain_A"][truncate_index:truncate_index+self.max_length]
                item['chain_mask'] = ([1]*len(self.data))[truncate_index:truncate_index+self.max_length]
                item['chain_encoding'] = ([1]*len(self.data))[truncate_index:truncate_index+self.max_length]
                item["ptm"]=item["ptm"][truncate_index:truncate_index+self.max_length]
        return item