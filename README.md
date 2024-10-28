BMAE model description:
1、Download the 110M-pretrained model.zip in the master branch.
2、Open main_finetune.py, add the .pth of 110M-pretrained model.zip to ‘--finetune’.
3, the header of the input data needs to contain the following information: for ‘timestamp’, ‘volt’, ‘current_C’, ‘cap’, ‘cycle’, ‘temp’, ‘quantity_C’;
4, The input data used for this model is charging data, which takes at least 147s.
5、Change ‘--data_path’ to your own data address and ‘--label_path’ to your own label location.
6、Input data format and label format can be seen in the data example folder.
