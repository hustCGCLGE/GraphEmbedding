from torch.utils.data import Dataset, DataLoader
from data.PreprocessData import ReadFileFromBin
import torch
import numpy as np
import os


class SDNEData(Dataset):
    def __init__(self, root, file_name, beta):
        super(SDNEData, self).__init__()
        data = ReadFileFromBin(root, file_name)

        self.beta = beta
        self.len = data.edge_num
        self.node_num = data.node_num
        self.offset = data.offset
        self.row_indices = data.row_indices
        self.column_indices = data.column_indices
        # print(self.row_indices)
        # print(self.column_indices)

    def __len__(self):
        return self.len

    def __getitem__(self, x):
        # produce the adjacent list of x1 and x2
        x1 = np.zeros(self.node_num, dtype=np.float32)
        x2 = np.zeros(self.node_num, dtype=np.float32)

        x1_pos = self.row_indices[x]
        x2_pos = self.column_indices[x]

        x1_pos = np.fromiter(self.column_indices[int(self.offset[x1_pos]):int(self.offset[x1_pos+1])], int)
        x2_pos = np.fromiter(self.column_indices[int(self.offset[x2_pos]):int(self.offset[x2_pos+1])], int)
        # print(x1_pos)
        # print(x2_pos)
        x1[x1_pos] = float(1)
        x2[x2_pos] = float(1)
        # print(x1)
        # print(x2)
        a1 = np.ones(self.node_num+1, dtype=np.float32)
        a2 = np.ones(self.node_num+1, dtype=np.float32)
        a1[x1_pos] = self.beta
        a2[x2_pos] = self.beta

        deg_i = self.offset[self.row_indices[x]+1] - self.offset[self.row_indices[x]]
        deg_j = self.offset[self.column_indices[x]+1] - self.offset[self.column_indices[x]]

        # print(self.offset)

        a1[self.node_num] = deg_i
        a2[self.node_num] = deg_j
        x_ij = np.ones(1, dtype=np.float32)
        return torch.from_numpy(x1), torch.from_numpy(x2),\
               torch.from_numpy(a1), torch.from_numpy(a2), x_ij


if __name__ == '__main__':
    train_data = DataLoader(SDNEData('./train', 'test.txt', 1), batch_size=1, shuffle=False, num_workers=4)
    for ii, (x1, x2, a1, a2, X_ij) in enumerate(train_data):
        print(ii)
        print(x1.dtype)
