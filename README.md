# Age and gender estimation based on Convolutional Neural Network and TensorFlow

---

This is a TensorFlow implement of face age and gender estimation which first using dlib to detect and align faces in the picture and then using a deep CNN to estimate age and gender.As you can see below,this project can estimate more than one face in a picture at one time.

![demo2](https://raw.githubusercontent.com/BoyuanJiang/Age-Gender-Estimate-TF/master/demo/demo2.jpg)

## Dependencies
This project has following dependencies and tested under CentOS7 with Python 3.6

- tensorflow==1.4
- dlib==19.7.99
- cv2
- matplotlib==2.1.0
- imutils==0.4.3
- numpy==1.13.3
- pandas==0.20.3


## Usage
### Make tfrecords
In order to train your own models,you should first download [imdb](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) or [wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) dataset,and then extract it under **data** path,after that,images path should look like
> /path/to/project/data/imdb_crop/00/somepictures  
/path/to/project/data/imdb_crop/01/somepictures  
....  
/path/to/project/data/imdb_crop/99/somepictures

Then you can run 
```bash
python convert_to_records_multiCPU.py --imdb --nworks 8
```
to convert images to tfrecords.**--imdb** means using imdb dataset,**--nworks 8** means using 8 cpu cores to convert the dataset parallelly.Because we will first detect and align faces in the pictures,which is a time consuming step,so we recommend to use as many cores as possible.Intel E5-2667 v4 and with 32 cores need approximately 50 minutes.

### Train model
Once you have converted images to tfrecords,you should have the following path:
> /path/to/project/data/train/train-000.tfrecords  
......  
/path/to/project/data/test/test-000.tfrecords  
......
 
 At present,our deep CNN uses FaceNet architecture,which based on inception-resnet-v1 to extract features.To speed up training,we use the pretrained model's weight from [this project](https://github.com/davidsandberg/facenet) and have converted the weight to adapt our model,you can download this converted pretrained facenet weight checkpoint from [here](https://drive.google.com/file/d/1ZT0oeJKh19gdcEUaoXGUOBjeMypaf550/view?usp=sharing). Extract it to path **models**.
 > /path/to/project/models/checkpoint  
 /path/to/project/models/model.ckpt-0.data-00000-of-00001  
 /path/to/project/models/model.ckpt-0.index  
 /path/to/project/models/model.ckpt-0.meta
 
 **NOTE:** This step is optional,you can also train your model from scratch.
 To start training,run
 
```bash
python train.py --lr 1e-3 --weight_decay 1e-5 --epoch 6 --batch_size 128 --keep_prob 0.8 --cuda
```
**NOTE:** Using the flag **--cuda** will train the model with GPU.

Using tensorboard to visualize learning
```
tensorboard --logdir=./train_log
```
### Test model
You can test all your trained models on testset through
```
python test.py --images "./data/test" --model_path "./models" --batch_size 128 --choose_best --cuda
```
Flag **--cuda** means using GPU when testing.**--choose_best** means testing all trained models and return the best one.If you just want to test the latest saved model,without this flag.
```
python test.py --images "./data/test" --model_path "./models" --batch_size 128 --cuda
```

### One picture demo
If you just want to test the model on your own picture,run
```
python eval.py --I "./demo/demo.jpg" --M "./models/" --font_scale 1 --thickness 1
```
Flag **--I** tells where your picture is.If the text label too small or too large on the picture,you can use a different **--font_scale 1** and **--thickness 1** to adjust the text size and thickness.
We also provide a pretrained model,you can download from [here](https://drive.google.com/file/d/16s-Ku1FbeVZyUQhEsvlbTi9hfyhvVmGK/view?usp=sharing) and extract it to **models** path.

### Picture from web cam

First download the pretrained model from [here](https://drive.google.com/file/d/1df5h2_KczLajGF9ZavaYxH-cSpGv3CUb/view?usp=sharing) and extract to **models** path.
In order to get pictures from web cam, you may need to uninstall your cv2 and [install it from source](https://www.scivision.co/anaconda-python-opencv3/) if have problems when running the below command:
```bash
python demo.py 
```
