class TMETEcfg(object):
    LOSS='ECP'#'ECP'
    BATCH_SIZE=4
    DEVICE='cuda'
    TEST_IMAGE_PATH = r'E:\Files\TinyMind\public_test_data\public_test_data\\'
    CROP_PATH = r'E:\Files\TinyMind\train_data\multi_test_crop\\'
    CROP_PATH_COMPLEMENT = r'E:\Files\TinyMind\train_data\multi_test_crop_complement\\'
    PATH = r'E:\Files\TinyMind\train_data\\'
    DETECT_MODEL_NAME = 'FoveaBox_FPN_BN_3X3_4h_xywh.pkl'
    RECOGNIZE_MODEL_NAME='TMVGG13BN.pkl'#'optimal0.49613821138211384TMVGG13BNLSTM_2_32_512.pkl'
    DETECT_NUM_CLASS = 1  # +1
    RECOGNIZE_NUM_CLASS=36
    TUPLE_L = (3, 5)  # (4,4)
    DETECT_EXPECTED_IMG_SIZE = (480, 240)
    RECOGNIZE_EXPECTED_IMG_SIZE=(200,32)
    MULTISCALE_SIZE = ((500, 250), (600, 300), (800, 400))
    DICTIONARY_NAME = 'dictionary_inv.json'
    OUTPUT_PATH='./submission/'
    def __init__(self):
        super(TMETEcfg,self).__init__()
        assert self.LOSS in ('CTC', 'ECP'), 'self.LOSS must be \'CTC\' or \'ECP\',but got :{}'.format(self.LOSS)
        if self.LOSS=='ECP':
            self.RECOGNIZE_EXPECTED_IMG_SIZE=(160,32)
            self.RECOGNIZE_NUM_CLASS=35

