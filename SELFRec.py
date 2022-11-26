from data.loader import FileIO
from util.helper import composePath

class SELFRec(object):
    def __init__(self, config):
        self.config = config
        # 图模型模型
        self.training_data = []
        self.test_data = []
        self.load_dataset()
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.' + self.config['type'] + '.' + \
            self.config['name'] + ' import ' + self.config['name']
        exec(import_str)
        recommender = self.config['name'] + \
            '(self.config,self.training_data,self.test_data)'
        eval(recommender).execute()

    def load_dataset(self):
        train_data_path = composePath(
            './dataset', self.config['dataset'], 'train.txt')
        test_data_path = composePath(
            './dataset', self.config['dataset'], 'test.txt')
        self.training_data = FileIO.load_data_set(
            train_data_path,
            self.config['type']
        )
        self.test_data = FileIO.load_data_set(
            test_data_path,
            self.config['type']
        )
