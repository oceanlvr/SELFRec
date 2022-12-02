from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys
from picture.feature import plot_features
import wandb

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        # 这里是初始化数据集相关的东西包括 user 和 item 总数量，ui_adj norm_adj 矩阵等
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(max(self.config['ranking']), candidates) #WIP
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.config['output']
        file_name = self.config['name'] + '@' + current_time + '-top-' + str(max(self.config['ranking'])) + 'items' + '.txt' # 
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['name'] + '@' + current_time + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, [int(num) for num in self.config['ranking']])
        wandb.log({ 'finnal_result': self.result })
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.config['name'], ''.join(self.result)))

    def drawPicture(self, emb, epoch):
        plot_features(emb, 'epoch:' + str(epoch) + ' ' + self.config['name'])

    def addBestPerformance(self, performance, key, value):
        performance[key] = value
    
    def addUserEmbedding(self, performance):
        self.addBestPerformance(performance, 'user_emb', self.user_emb.cpu())

    def fast_evaluation(self, epoch):
        print('evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [max(self.config['ranking'])])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0: # 如果当前的性能比之前的好，就更新最好的性能。这里的好是指好的指标的数量比差的指标多
                self.bestPerformance[0] = epoch + 1
                self.addUserEmbedding(performance)
                self.bestPerformance[1] = performance
                self.save()
        else: # 没有最好的结果直接更新
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.addUserEmbedding(performance)
            self.bestPerformance.append(performance)
            self.save()
        self.drawPicture(self.bestPerformance[1]['user_emb'], epoch)
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(max(self.config['ranking'])) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        bp = ''
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + ' | '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + ' | '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + ' | '
        bp += 'MDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        wandb.log({ 
            'hit_ratio': self.bestPerformance[1]['Hit Ratio'],
            'precision': self.bestPerformance[1]['Precision'],
            'recall': self.bestPerformance[1]['Recall'],
            'NDCG': self.bestPerformance[1]['NDCG'],
            'target': 0.1*self.bestPerformance[1]['Hit Ratio'] +
                0.1*self.bestPerformance[1]['Precision'] + 
                0.4*self.bestPerformance[1]['Recall']+
                0.4*self.bestPerformance[1]['NDCG']
        })
        # bestPerformance

        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure
