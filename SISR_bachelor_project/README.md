# Single Image Super-Resolution using SRCNN and SRGAN
Tensorflow implementation of Convolutional Neural Networks for super-resolution and Generative adversarial Neural Networks. 

## Dependencies

- Python 3
- Tensorflow
- Skimage
- Numpy
- Matplotlib
- OpenCV2

## How to Test both SRCNN and SRGAN networks

1. Clone this github repo. 
```
git clone https://github.com/teouw/SISR_bachelor_project.git
cd SRCNN or cd SRGAN
```
2. Place your own **high-resolution testing images** in `./input` folder.
3. Run test.
```
python3 test.py
```

4. The results are in `./output` folder.
 
## How to Train both SRCNN and SRGAN networks

1. Clone this github repo. 
```
git clone https://github.com/teouw/SISR_bachelor_project.git
cd SRCNN or cd SRGAN
```
2. Place your own **high-resolution training images** in `./data` folder.
3. Run training.
```
python3 train.py
```
4. The trained model will be safe in `./model` folder.

## Authors

- [Kaltrachian Téo](https://github.com/teouw)
