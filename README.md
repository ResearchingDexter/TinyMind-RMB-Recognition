# TinyMind-RMB-Recognition
The code is for the competition [RMB code recognition](https://www.tinymind.cn/competitions/47#overview "RMB code recognition")
and my rank is 13th in the final 
# Docs
I used **foveabox** to detect the object box and **crnn** to recognize the RMB code,so I trained the detection model and the recognition model respectively. And cascade them in the test time.

```python
-TMDetector  #the part of detection(i.e. foveabox)
 -TMBackbone #the backbone 
 -TMDetectConfigure  #the configure file for train and test
 -TMDetectDataSet    #the dataset
 -TMDetectLoss       #the loss
 -TMDetectUtils      #generate train target for foveabox
 -TMRPN              #the rpn head
 -TMTest             #the test code of the detection part of the test
 -TMTrain            #the train code of the detection part of the train
#the part of crnn
-TMTextLine
 -TMNonLocalNet      #the NonLocal Modular
 -TMTextLineConfigure#the  configure file for train and test
 -TMTextLineDataSet  #the dataset for crnn
 -TMTextLineNN       #the model for crnn
 -TMTextLineTest     #the test code for crnn
 -TMTextLineTrain    #the train code for crnn
-TMEndToEndConfigure #the end-to-end configure file while includes detection and recognition
-TMEndToEndTest      #the end-to-end test code which includes detection and recognition
```
