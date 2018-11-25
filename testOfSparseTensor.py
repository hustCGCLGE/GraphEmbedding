import torch as t
import numpy as np
import time
import torch.nn as nn


class SParseLinear(nn.Module):
    def __init__(self, in_feature, out_feature1, out_feature2):
        super(SParseLinear, self).__init__()
        self.fc = nn.Linear(in_feature, out_feature1, bias=False)
        self.fc1 = nn.Linear(out_feature1, out_feature2, bias=False)
        self.fc2 = nn.Linear(out_feature2, out_feature1, bias=False)
        self.fc3 = nn.Linear(out_feature1, in_feature, bias=False)

    def forward(self, x):
        t.cuda.synchronize()
        start = time.time()

        x = self.fc1(self.fc(x))
        t.cuda.synchronize()
        e1 = time.time()
        print('encoder time is %.06f' % (e1- start))
        x = self.fc3(self.fc2(x))
        print('decoder time is %.06f' % (time.time() - e1))

        return x


if __name__ == '__main__':
    # model
    in_f = 48
    out_f = 48
    out_f1 = 20

    model = SParseLinear(in_f, out_f, out_f1)

    # data
    input = []

    v_num = 1000
    batch_size = 1
    row_pos =[]
    column_pos = []
    for i in range(batch_size):

        row_pos = np.append(row_pos, np.ones(v_num) * i)
        column_pos = np.append(column_pos, np.random.random_integers(0, in_f-1, v_num))
    row_pos = t.Tensor(row_pos).view(1, -1).long()
    column_pos = t.Tensor(column_pos).view(1, -1).long()

    i = t.cat((row_pos, column_pos), dim=0)
    v = t.ones(v_num * batch_size)

    x = t.sparse.FloatTensor(i, v, t.Size([batch_size, in_f]))

    # t.cuda.set_device(3)
    i.cuda()
    v.cuda()
    x = x.cuda()

    x = t.randn(in_f)
    x = x.cuda()
    model.cuda()

    model(x)
    model(x)

