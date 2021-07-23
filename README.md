## Directory Structure 
```
$root
.
├── README.md
├── requirements.txt
├── dataset.py
├── test.py
├── train.py
├── data
│     ├── test
│     └── train
└── weights
    └── best_weights.pth
```

## Steps to run
1. Make sure you have all the dependencies installed from [requirements.txt](requirements.txt).
2. Create a [weights](weights) in the ```root``` directory and download pre-trained weight files [best_weights.pth](https://drive.google.com/file/d/1NGpx2WyApqjWuIHmkEVyLz7a9j_ossZ0/view?usp=sharing) to it.
3. Run [train.py](train.py) to train the model.
4. Run [test.py](test.py) to test the model on the test dataset.