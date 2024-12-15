# -*- coding: UTF-8 -*-
import os
import logging
import numpy as np
import pandas as pd
from utils import utils


class myReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep # sep of csv file
        self.prefix = args.path # data dir的路径
        self.dataset = args.dataset # dataset文件夹的名字
        self._read_data() # 读取数据

        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            # 不用再按照time排序，因为数据集预处理已经排过了
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True)
            if 'time' in self.data_df[key].columns: # DiffRec提供的数据集没有time列，但是是已经按照时间排序的
                self.data_df[key] = self.data_df[key].sort_values(by = ['user_id','time'])
            else:
                self.data_df[key] = self.data_df[key].sort_values(by = ['user_id'])
            self.data_df[key] = utils.eval_list_columns(self.data_df[key]) #有一些列表值是list，直接读取进来就是str，这里转换回list

        logging.info('Counting dataset statistics...')
        if 'time' in self.data_df['train'].columns:
            key_columns = ['user_id','item_id','time']
        else:
            key_columns = ['user_id','item_id'] # DiffRec提供的数据没有time

        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction 用不着，DiffRec没有CTR任务
            key_columns.append('label')
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        # DiffRec的id是从0开始的，框架原本是从1开始的，我们统一也从0开始
        # self.n_users是用户数，self.n_items是物品数
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users, self.n_items, len(self.all_df)))
        # DiffRec无label
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))
        
