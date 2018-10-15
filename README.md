# attribute

This is a repository on attribute prediction.
Training is done using CelebA dataset or DeepFashion dataset.

## Using CelebA dataset
```
python main.py
```

optional arguments

    num_epoch : number of epochs to train
    attrnum : number of attributes
    learning_rate : initial learning rate of Adam optimizer
    batch_size : batch size for training
    root_dir : full path of data. should be parent for images and annotations
    img_dir : relative path of directory containing images
    ann_dir : relative path of file containing annotations of attributes 

