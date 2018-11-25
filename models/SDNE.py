import torch
import torch.sparse as ts
import torch.nn as nn
import numpy as np
import time


class SDNE(nn.Module):

    def __init__(self,  num_units, k, d):
        super(SDNE, self).__init__()

        # auto_encoder
        auto_encoder = list()
        auto_encoder.append(nn.Linear(num_units[0], num_units[1]))
        # auto_encoder.append(nn.Dropout(0.5))
        for i in np.arange(1, k):
            auto_encoder.append(nn.Linear(num_units[i], num_units[i+1]))
        auto_encoder.append(nn.Linear(num_units[k], d))

        self.auto_encoder = nn.Sequential(*auto_encoder)

        # auto_decoder
        auto_decoder = list()
        auto_decoder.append(nn.Linear(d, num_units[k]))
        auto_decoder.append(nn.Dropout(0.2))
        for i in np.arange(0, k):
            auto_decoder.append(nn.Linear(num_units[k - i], num_units[k-i-1]))

        self.auto_encoder = nn.Sequential(*auto_encoder)
        self.auto_decoder = nn.Sequential(*auto_decoder)

    def forward(self, x):
        start = time.time()
        # print(x.dim())
        # y = x
        # print(type(x))
        # for m in self.auto_encoder:
        #     start = time.time()
        #     print(m)
        #     y = m(y)
        #
        #     torch.cuda.synchronize()
        #     print(type(y))
        #     print(time.time() -start)
        # t = y
        # for m in self.auto_decoder:
        #     start = time.time()
        #     print(m)
        #     t = m(t)
        #     torch.cuda.synchronize()
        #     print(type(t))
        #     print(time.time() -start)
        # x_hat = t
        y = self.auto_encoder(x)
        torch.cuda.synchronize()

        end_time = time.time()
        print("encoder time : " + str(time.time() - start))
        x_hat = self.auto_decoder(y)
        torch.cuda.synchronize()

        print("decoder time: " + str(time.time() - end_time))
        return x - x_hat, y


if __name__ == '__main__':

    torch.cuda.set_device(3)
    node_num = 4847571
    edge_node = 100
    rand_pos = np.random.random_integers(0, node_num, edge_node)
    input = np.zeros(node_num, np.float32)
    print(input.dtype)

    num_units = list()
    num_units.append(node_num)
    for i in range(1):
        num_units.append(int(node_num / 100000))
    d = 20

    model = SDNE(num_units, 1, d)

    # for name, para in model.named_parameters():
    #     print(name)
    #     print(para.dtype)
    #     print(para.size())

    for t in rand_pos:
        input[t] = float(1)

    t_rand_pos = torch.from_numpy(rand_pos).long()
    row_indices = torch.zeros(edge_node).long()
    indices = torch.cat((row_indices.view(1, -1), t_rand_pos.view(1, -1)), dim=0)
    v = torch.ones(indices.size()[-1])
    size = torch.Size([1, node_num])

    print(indices)
    print(v)
    print(size)
    x = ts.FloatTensor(indices, v, size)

    print(model)
    if torch.cuda.is_available():
        # sparse tensor do not support .cuda()
        print(edge_node)
        indices.cuda()
        v.cuda()
        input = x.cuda()
        print(input.dtype)
        print(type(x.cuda()))
        model.cuda()
        print(model(input))



    #     x.cuda(device=3)
    #     model.cuda(device=3)
    #     print(x.size())
    # x_hat, y = model(x)
