from config import DefaultConfig
import models
from torch.utils.data import DataLoader
import torch.optim
from data import SDNEData
from LossFunc import MyLoss
import time
import fire


def train(**kwargs):
    # step1 : customize config
    opt = DefaultConfig()
    opt.parse(kwargs)

    # step2 : model

    # TODO: make the initiation correspond to the data

    # Problem : when node_num is greater than 5M, the depth of encoder hidden layers can not be greater than 3
    #           the number of parameters will be very large


    print('Initiate model')
    model = getattr(models, opt.model)(opt.num_units, 1, opt.d)

    print(model)

    # if opt.load_model_path:
    #     model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    model.train()
    print('Initiate Train_Data')
    # step3 : prepare data. In this case, only training data is needed
    train_data = DataLoader(SDNEData(opt.train_data_root, opt.file_name, opt.beta),
                            shuffle=True,
                            num_workers=1,
                            pin_memory=False)

    print('Initiate Optimizer and Loss function')
    # step4 : optimizer
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.99, nesterov=True, weight_decay=1e-5)
    loss_func = MyLoss(1, 1, 1e-5, opt.node_num).cuda()
    total_time = 0.0

    for epoch in range(0, opt.max_epoch):

        epoch_st_time = time.time()
        # Dimension: x1,x2, num_node ; a1,a2, num_node+1 , X_ij, num_node
        for ii, (x1, x2, a1, a2, x_ij) in enumerate(train_data):
            if ii == 10:
                break

            if opt.use_gpu:

                x1 = x1.cuda()
                x2 = x2.cuda()
                a1 = a1.cuda()
                a2 = a2.cuda()
                x_ij = x_ij.cuda()
            x_diff1, y1 = model(x1)
            x_diff2, y2 = model(x2)
            y_diff = y2 - y1

            optimizer.zero_grad()
            loss = loss_func(x_diff1, x_diff2, y_diff, a1, a2, x_ij)

            # print(torch.cuda.memory_allocated())
            # print(torch.cuda.max_memory_allocated())
            # # if(ii%4000000 == 0):
            #     print(loss)
            start_time = time.time()
            loss.backward(retain_graph=False)
            torch.cuda.synchronize()
            end_time = time.time()
            print('backward time = ' + str(end_time - start_time))
            optimizer.step()

        epoch_end_time = time.time()
        total_time += epoch_end_time - epoch_st_time
    print("total_time is " + str(total_time) + 's\n')


if __name__ == '__main__':
    fire.Fire()
