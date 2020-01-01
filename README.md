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
 -TMRPN              #the rpn
 -TMTest
 -TMTrain
#the part of crnn
-TMTextLine
 -TMTextLineDataSet
 -TMTextLineNN
 -TMTextLineTest     
 -TMTextLineTrain
```
