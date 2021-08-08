import argparse

class ArgParser:
    def __init__(self, args) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument('-m',
                            action='store', dest='model',
                            type=str,
                            required=True,
                            help='select type of model to use [svm/cnn]')
        parser.add_argument('-t',
                            action='store', dest='train',
                            type=str,
                            required=True,
                            help='train new model [True/False]')
        parser.add_argument('-n',
                            action='store', dest='num',
                            type=int,
                            required=True,
                            help='number of tweets to look for')
        parser.add_argument('-d',
                            action='store', dest='daysBack',
                            type=int,
                            required=True,
                            help='maximum number of days before today to search through')

        self.CONFIG = parser.parse_args(args[1:])
    
    def get_train(self) -> bool:
        if self.CONFIG.train.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        elif self.CONFIG.train.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    def get_daysBack(self) -> int:
        return self.CONFIG.daysBack
    
    def get_num_tweets(self) -> int:
        return self.CONFIG.num

    def get_model_name(self) -> str:
        return self.CONFIG.model