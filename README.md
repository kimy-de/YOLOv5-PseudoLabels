## YOLOv5 with Pseudo Labels
This code is a modified version based on YOLOv5 provided by the repository(https://github.com/ultralytics/yolov5). The purpose of this repository is to augment train data by pseudo labels so this repository contains a generator of pseudo labels not provided by the original code. 


<img src="https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png" width="1000">

## Tutorial
If you want to learn about training, testing, customizing, and so on, please refer to the original repository.
* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

## Generator of pseudo labels
Before using the generator, prepare a pretrained model. The generator makes pseudo labels and their images based on your model. 
Next, make "extra" in "images" folder, and "pseudo" folders in "images" and "labels" directories. "extra" directory includes unlabelled images. "pseudo" is a folder to save pseudo labels and their images.

```
# custom.yaml
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../coco128/images/train2017/train/
val: ../coco128/images/train2017/valid/
pseudo: ../coco128/images/train2017/pseudo/ # Selected images from the extra folder for pseudo labels
extra: ../coco128/images/train2017/extra/ # Prepared images for pseudo labels(candidate group for pseudo labels)
```

Finally, use the code with a threshold which means the lower bound of class confidence in [0,1] range.
```bash
$ python pseudolabel.py --data custom.yaml --pseudo_threshold 0.4 --weights 'yolo5s.pt'                                   
```

## Training with pseudo labels
If --pseudo False, the code is running with no difference from the original code. Otherwise, a model uses both train and pseudo labels for training.
```bash
$ python train.py --data custom.yaml --cfg yolov5s.yaml --weights '' --batch-size 64 --pseudo True                                    
```

## To Do
- [ ] Modification of the loss function
- [ ] Scheduling of a loss penalty for pseudo labels
