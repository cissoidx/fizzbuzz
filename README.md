# fizzbuzz
fizzbuzz implementation in pytorch

Check [this link](https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/) out to find what fizzbuzz is and what this repo is for.

### datasets
train range: 101 ~ 1023  
test range: 1 ~ 100

### train
To train fizzbuzz, simply run  
```bash
cd $fizzbuzz_root
python trainfizzbuzz.py
```

Uncomment the lines with cuda to run on GPU. Hyper parameters are tuned already to get 100% acc on test dataset.
