# -*- coding: UTF-8 -*-
import torch
import numpy as np
import scipy.sparse as sp
from my_new_models.DiffRec import *


# T-DiffRec只在数据处理上有所不同，因此我们只需要重写Dataset类即可。
class TDiffRec(DiffRec):
	reader, runner = 'myReader', 'myRunner' # 使用框架自带的序列数据读取器，帮助我们排序
	extra_log_args = {
		'batch_size': 'bs',
		'dims': 'dims',
		'emb_size': 'embs',
		'steps': 'st',
		'noise_scale': 'ns',
		'noise_min': 'nmin',
		'noise_max': 'nmax',
		'sampling_steps': 'sample_st',
		'reweight': 'rw'
	}
	@staticmethod
	def parse_model_args(parser):
		# T-DiffRec的参数
		parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
		parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')
		return DiffRec.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.w_min=args.w_min
		self.w_max=args.w_max


	class Dataset(DiffRec.Dataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict() # 如果设定了要缓存则除了训练集，其他集合的数据都会缓存，将通过_get_feed_dict得到的数据进行缓存
			self.data = corpus.data_df[phase].to_dict('list') #是字典
			# ↑ DataFrame is not compatible with multi-thread operations

			if self.phase != "train":
				self.csr_matrix = sp.csr_matrix((np.ones_like(corpus.data_df[phase]['user_id']), (corpus.data_df[phase]['user_id'], corpus.data_df[phase]['item_id'])),
										dtype='float32', shape=(self.corpus.n_users, self.corpus.n_items))
				self.csr_matrix_A = torch.FloatTensor(self.csr_matrix.A)
			else:
				csr_matrix_ori = sp.csr_matrix(
					(np.ones_like(corpus.data_df[phase]['user_id']),
					(corpus.data_df[phase]['user_id'], corpus.data_df[phase]['item_id'])),
					dtype='float32',
					shape=(self.corpus.n_users, self.corpus.n_items)
				)
				counts = np.diff(csr_matrix_ori.indptr)
				weights = np.hstack([np.linspace(self.model.w_min, self.model.w_max, count) for count in counts])
				self.csr_matrix = sp.csr_matrix(
					(weights, csr_matrix_ori.indices, csr_matrix_ori.indptr),
					dtype='float32',
					shape=csr_matrix_ori.shape
				)
				self.csr_matrix_A = torch.FloatTensor(self.csr_matrix.A)
		
