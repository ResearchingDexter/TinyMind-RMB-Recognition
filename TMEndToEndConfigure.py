class TMETEcfg(object):
    LOSS='CTC'#'ECP'
    BATCH_SIZE=4
    DEVICE='cuda'
    TEST_IMAGE_PATH =r'E:\Files\TinyMind\private_test_data\private_test_data\\' #r'E:\Files\TinyMind\public_test_data\public_test_data\\'
    CROP_PATH = r'E:\Files\TinyMind\train_data\multi_test_crop\\'
    CROP_PATH_COMPLEMENT = r'E:\Files\TinyMind\train_data\multi_test_crop_complement\\'
    PATH = r'E:\Files\TinyMind\train_data\\'
    DETECT_MODEL_NAME = 'FoveaBox_FPN_BN_3X3_4h_xywh.pkl'
    RECOGNIZE_MODEL_NAME='optimal0.49613821138211384TMVGG13BNLSTM_2_32_512.pkl'#'TMNLVGG19BNLSTM_2_32_512.pkl:0.385452'#'DenseLSTM_2_32_512.pkl'
    #'optimal0.9971047331319235TMVGG19BNLSTM_2_32_512.pkl'#'DenseLSTM_2_32_512.pkl'#'optimal0.49613821138211384TMVGG13BNLSTM_2_32_512.pkl':0.9785
    DETECT_NUM_CLASS = 1  # +1
    RECOGNIZE_NUM_CLASS=36
    TUPLE_L = (3, 5)  # (4,4)
    DETECT_EXPECTED_IMG_SIZE = (480, 240)
    RECOGNIZE_EXPECTED_IMG_SIZE=(200,32)
    MULTISCALE_SIZE = ((400,200),(420,210),(448,224),(500, 250),(540,270), (600, 300), (640,320),(800, 400))
    DICTIONARY_NAME = 'dictionary_inv.json'
    OUTPUT_PATH='./submission/'
    def __init__(self):
        super(TMETEcfg,self).__init__()
        assert self.LOSS in ('CTC', 'ECP'), 'self.LOSS must be \'CTC\' or \'ECP\',but got :{}'.format(self.LOSS)
        if self.LOSS=='ECP':
            self.RECOGNIZE_EXPECTED_IMG_SIZE=(160,32)
            self.RECOGNIZE_NUM_CLASS=35
            self.RECOGNIZE_MODEL_NAME='SEDenseFC.pkl'#'optimal0.9970996978851964DenseFC.pkl'#'DenseFC.pkl'#:1x1_0.97.2#'TMVGG13BN.pkl'

