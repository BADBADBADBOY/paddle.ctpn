# paddle.ctpn
PaddlePaddle 复现 Detecting Text in Natural Image with Connectionist Text Proposal Network


效果：

原论文在3000张图片上训练，而本项目只在icdar2015（1000张）图像上训练，这里提供两个版本，ctpn版实现几乎与原论文没有出入，ctpn_tail几乎没有引人额外开销，算是对icdar2015这样检测单个单词的一个简单优化版。

|method|precision|recall|Hmean
|-|-|-|-|
|原论文|74.22% |51.56%|60.85%|																																																						
|ctpn|55.66%|40.68%|47.00%|
|ctpn_tail|64.48%|56.91%|60.46%|

### 写在前面

根目录需要建4个文件夹：result, pre_gt, pre_model, model_save 用来放验证时结果图片，验证结果txt文件，预训练模型，训练模型保存

### 环境
这里用的是aistudio，paddlepaddle-2.1.2
这里提供了三个版本的vgg，一个是我torch转的,也是我训练用的[下载地址](https://pan.baidu.com/s/1ut4VeSJw6w5AKsuC62WRlw)(提取码:fxw6),下载好了，放在pre_model文件夹，第二个是paddlecls里面的，第三个paddle vision里的，可在models/ctpn.py自己打开注释更换。
这里提供了俩种的bestmodel[下载地址](https://pan.baidu.com/s/1ut4VeSJw6w5AKsuC62WRlw)(提取码:fxw6)

需要编译下nms和计算iou的函数
```
cd utils/bbox
sh make.sh
```


### 参数说明
|参数|类型|说明|
|-|-|-|
|optimizer|str|优化器，建议SGD
|val_dir|str|验证集文件夹地址
|val_gt_path|str|验证集gt文件地址
|batch_size|int|训练的batch
|restore|bool|中断时是否恢复训练
|restore_epoch|int|从第几个epoch恢复
|size_list|list|本来想多尺度训练，奈何paddle dataload有点不支持，暂时默认
|num_worker|int|dataload worker 数
|lr|int|学习率
|step_size|int|学习率多少epoch调整1次
|gamma|float|调整学习率的尺度
|train_epochs|int|训练多少个epoch
|start_val|int|从多少个epoch开始做验证
|show_step|int|多少次step显示一次loss
|epoch_save|int|多少epoch保存一次模型
|checkpoint|str|模型保存的地址



### 训练

设置训练图片的路径，在这里20-22行[位置](./dataLoader/dataLoad.py)

```
python3 train.py --batch_size 8 --lr 0.08 --val_dir /src/icdar2015/test_img --val_gt_path /src/icdar2015/test_gt
```

 设置训练图片的路径，在这里20-22行[位置](./dataLoader/dataLoad_tail.py)

```
python3 train_tail.py --batch_size 8 --lr 0.08 --val_dir /src/icdar2015/test_img --val_gt_path /src/icdar2015/test_gt
```

### 断点恢复训练

假设保存了一个模型是ctpn_12.pdparams

```
python3 train.py --batch_size 8 --lr 0.08 --val_dir /src/icdar2015/test_img --val_gt_path /src/icdar2015/test_gt --restore True --restore_epoch 12
```

```
python3 train_tail.py --batch_size 8 --lr 0.08 --val_dir /src/icdar2015/test_img --val_gt_path /src/icdar2015/test_gt --restore True --restore_epoch 12
```

### infer

进文件设置好路径后，运行：

```
python3 inference.py 
```

```
python3 inference_tail.py 
```
