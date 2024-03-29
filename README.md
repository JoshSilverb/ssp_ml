# Stock Sentiment Prediction (SSP)

Analyze sentiment (bearish/bullish/neutral) of tweets containing a given ticker using a linear SVM model or convolutional neural network


## Setup tutorial

### Requirements:
- python pip (install with ```sudo apt install python3-pip```)
- python 3 (install with ```sudo apt install python```)
- Packages listed in requirements.txt (install with ```pip install -r requirements.txt```)

### Running
To run the basic version, simply run ```python main.py``` and enter a ticker when prompted
To run with additional configuration, run ```python main.py -m [svm/cnn] -t [True/False] -n [Integer] -d [Integer]```, where 
- ```-m``` selects the type of model to use for classification (SVM or CNN)
- ```-t``` selects whether to retrain the model using data in ```./data/train/```
- ```-n``` defines the number of tweets to analyze
- ```-d``` defines the maximum number of days before today to search through


