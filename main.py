from utils.argParser import ArgParser
from utils.driver import Driver

import sys
from os import get_terminal_size


            
if __name__ == "__main__":
    
    '''
    API_key = input("API key: ")
    API_secret = input("API key secret: ")
    access_token = input("Access token: ")
    access_secret = input("Access token secret: ")
    
    system('clear')
    '''

    API_key = "UjZiYFkC1UJFHDVUHdCikJQYG"
    API_secret = "PFvAZfR51zNVK03ipLWpyBOjJCKjibRx1YvSR9Am5LplyIWunt"
    access_token = "2911122124-XpSTbAnFvYmIuSPaZMvXVvFrmCvMNNSxGXs3sfU"
    access_secret = "uHdAiCoQPkV6Ng9vv0sHXadTDF19PdCYjhFKOsWkQw0V7"
    
    
    

    if len(sys.argv) > 1:
        parser = ArgParser(sys.argv)
        driver = Driver(parser.get_model_name(), API_key, API_secret, access_token, access_secret, parser.get_train(), parser.get_daysBack(), parser.get_num_tweets())
    else:
        # default setup
        print("Default setup")
        driver = Driver("cnn", API_key, API_secret, access_token, access_secret, False, 2, 50)
    
    col, l = get_terminal_size()
    print('='*col)
    print("Stock Sentiment Predictor - Josh Silverberg - https://github.com/JoshSilverb/ssp_ml")
    print("Type \"help\" for help")

    ticker = input(">> Stock ticker name: ")
    while len(ticker) > 0:
        if ticker == "help":
            print("\t================ help ================")
            print("\t>> Stock ticker name: [ticker name for stock to investigate, eg TSLA]")
            print("\t======================================")
            ticker = input(">> Stock ticker name: ")
            continue

        label, percent = driver.classify(ticker)
        
        if label == "invalid":
            print("No tweets found - likely invalid ticker")
        else:
            print("Predicted:", label, "with", str(int(percent*100)) + "% confidence")


        ticker = input(">> Stock ticker name: ")


