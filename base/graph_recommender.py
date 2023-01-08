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
        super(GraphRecommender, self).__init__(
            conf, training_set, test_set, **kwargs)
        # 这里是初始化数据集相关的东西包括 user 和 item 总数量，ui_adj norm_adj 矩阵等
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = {
            'epoch': -1,
            'metric': {},
            'addon': {},
            'hasRecord': False
        }

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (
            self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (
            self.data.test_size()))
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
            r = '\rProgress: [{}{}]{}%'.format(
                '+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
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
            ids, scores = find_k_largest(
                max(self.config['ranking']), candidates)  # WIP
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append(
            'userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
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
        file_name = self.config['name'] + '@' + current_time + \
            '-top-' + str(max(self.config['ranking'])) + 'items' + '.txt'
        FileIO.write_file(out_dir, file_name, self.recOutput)
        print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['name'] + '@' + \
            current_time + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, [
                                         int(num) for num in self.config['ranking']])
        wandb.log({'finnal_result': self.result})
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' %
              (self.config['name'], ''.join(self.result)))

    def addPerformanceAddon(self):
        performance = {}
        performance['user_emb'] = self.model.embedding_dict['user_emb'].detach(
        ).cpu().numpy()
        performance['item_emb'] = self.model.embedding_dict['item_emb'].detach(
        ).cpu().numpy()
        return performance

    def getMetric(self, measure):
        metric = {}
        for m in measure[1:]:
            k, v = m.strip().split(':')
            metric[k] = float(v)
        return metric

    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [
                                     max(self.config['ranking'])])
        metric = self.getMetric(measure)

        # 如果当前的性能比之前的好，就更新最好的性能。这里的好是指好的指标的数量比差的指标多
        count = 0
        for k in self.bestPerformance['metric']:
            if self.bestPerformance['metric'][k] > metric[k]:
                count += 1
            else:
                count -= 1

        if (not self.bestPerformance['hasRecord']) or count > 0:
            addon = self.addPerformanceAddon()
            bestPerformance = {
                'epoch': epoch + 1,
                'metric': metric,
                'addon': addon,
                'hasRecord': True
            }
            self.bestPerformance = bestPerformance
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' +
              str(max(self.config['ranking'])) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', ' | '.join(measure))
        # set target
        wandb.log({
            'epoch': epoch + 1,
            'hit_ratio': metric['Hit Ratio'],
            'precision': metric['Precision'],
            'recall': metric['Recall'],
            'NDCG': metric['NDCG'],
            'target': 0.1*metric['Hit Ratio'] +
            0.1*metric['Precision'] +
            0.4*metric['Recall'] +
            0.4*metric['NDCG']
        })
        bp = ''
        curMetric = self.bestPerformance['metric']
        bp += 'Hit Ratio' + ':' + \
            str(curMetric['Hit Ratio']) + ' | '
        bp += 'Precision' + ':' + \
            str(curMetric['Precision']) + ' | '
        bp += 'Recall' + ':' + str(curMetric['Recall']) + ' | '
        bp += 'MDCG' + ':' + str(curMetric['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance['epoch']) + ',', bp)
        print('-' * 120)
        # print('Addon:', ',', str(self.bestPerformance['addon']))
        # print('-' * 120)
        # if (epoch + 1) % 10 ==0:
        #     self.drawheatmaps()
        return measure

    def afterTrain(self):
        self.drawheatmaps()

    def drawheatmaps(self):
        self.drawheatmap(self.bestPerformance['addon']['user_emb'], 'user_emb_'+str(self.bestPerformance['epoch']))
        self.drawheatmap(self.bestPerformance['addon']['item_emb'], 'item_emb_'+str(self.bestPerformance['epoch']))

    def drawheatmap(self, emb, name):
        plot_features(emb, self.config['name'] + '_' + name)
