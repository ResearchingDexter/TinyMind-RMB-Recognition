class TMcfg:
    PRETRAIN=True
    DEVICE='cuda'
    FPN=True
    BATCH_SIZE=1
    LR=0.01
    EPOCH=10000
    MAX_ACCURACY=0.5
    PATH=r'E:\Files\TinyMind\train_data\\'
    TEST_IMAGE_PATH=r'E:\Files\TinyMind\public_test_data\public_test_data\\'
    LABEL_PATH=r'E:\Files\TinyMind\train_data\train_label\\'
    CROP_PATH=r'E:\Files\TinyMind\train_data\multi_test_crop\\'
    #IMAGE_PATH=r'E:\Files\TinyMind\train_data\train_data\\'
    IMAGE_PATH=TEST_IMAGE_PATH
    IMAGE_NAME='1HQ0TNBZ.jpg'
    MODEL_PATH=r'E:\Files\TinyMind\train_data\\'
    MODEL_NAME='FoveaBox_FPN_BN_3X3_4h_xywh.pkl'#'FoveaBox.pkl'
    NUM_CLASS=1#+1
    DEVICE_ID=[1,2]#[1,2]
    DISTRIBUTED=False
    TUPLE_L=(3,5)#(4,4)
    TOTAL_NUM=1000
    POSTIVE_NUM=500
    EXPECTED_IMG_SIZE = (500,250)#(448, 224)(500,250)(600,300),(800,400)
    MULTISCALE_SIZE=((448,224),(500,250),(600,300),(640,320),(800,400))
if TMcfg.DISTRIBUTED==True:
    TMcfg.DEVICE+=':'+str(TMcfg.DEVICE_ID[0])