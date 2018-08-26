# Training
$cd code
$python train.py --model beenburger --save_dir DIR_NAME --data_name coco 2>&1 | tee ../log/logs.txt

# If files are not exist, download them by using below commands
# Download pretrained w2v embedding files
$wget -P ./w2v/ http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
$wget -P ./w2v/ http://www.cs.toronto.edu/~rkiros/models/utable.npy
$wget -P ./w2v/ http://www.cs.toronto.edu/~rkiros/models/btable.npy

# Make coco vocabulary
$cd code/preprocess
$python make_vocab.py

# Make coco id list
$cd code/preprocess
$python make_npy.py

# Download initial-pretrained resnet + 1x1 conv layers
https://drive.google.com/open?id=1ldRO9LzTg2_1HPlqA1flpK7T0QEGBmHM

