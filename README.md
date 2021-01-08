# Stock Sentiment Prediction (SSP)

Analyze sentiment (bearish/bullish) of tweets with given $ticker


## Setup tutorial

### Requirements:
- python pip (install with ```sudo apt install python3-pip```)
- python 3 (install with ```sudo apt install python```)
- Packages listed in requirements.txt (install with ```pip install -r requirements.txt```)

### Running
To run the basic version, simply run ```python main.py``` and enter a ticker when prompted
To run with additional configuration, run ```python main.py -t [True/False] -n [Integer] -d [Integer]```, where 
- ```-t``` selects whether to retrain the model using data in ```./data/train/```
- ```-n``` defines the number of tweets to analyze
- ```-d``` defines the maximum number of days before today to search through