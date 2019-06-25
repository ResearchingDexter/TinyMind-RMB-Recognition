class cfg(object):
    LOSS = 'CTC'  # 'ECP'
    DEVICE='cuda'
    BATCH_SIZE=8
    EPOCH=10000
    PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\\'
    DICTIONARY_NAME='RecTs2dictionary.json'
    IMAGE_PATH=r'E:\Files\TinyMind\train_data\\train_data\\'
    MODEL_PATH=r'E:\Files\TinyMind\train_data\\'
    MODEL_NAME='DenseCNN.pkl'
    EXPECTED_IMG_SIZE=(200,32)
    PRETRAIN=False
    NUM_CLASS=36
    LR=0.001
    MAX_ACCURACY=0
    def __init__(self):
        super(cfg,self).__init__()
        assert self.LOSS in ('CTC', 'ECP'), 'self.LOSS must be \'CTC\' or \'ECP\',but got :{}'.format(self.LOSS)
        if self.LOSS=='ECP':
            self.EXPECTED_IMG_SIZE=(160,32)