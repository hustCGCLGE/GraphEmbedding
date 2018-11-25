import warnings


class DefaultConfig:
    env = 'default'
    model = 'SDNE'
    train_data_root = './data/train'
    test_data_root = '/data/test'
    file_name = 'soc-LiveJournal1.txt'
    #load_model_path = 'checkpoints/model.pth'

    batch_size = 1
    use_gpu = True
    num_workers = 4
    print_freq = 20

    debug_file = '.'
    result_file = 'result.csv'

    max_epoch = 1
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4
    beta = 5

    node_num = 4847571

    num_units = list()
    num_units.append(node_num)
    num_units.append(int(node_num / 100000))
    d = 20

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s " % k)
            setattr(self, k, v)
        print('usr config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))