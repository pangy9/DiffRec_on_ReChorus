# -*- coding: UTF-8 -*-
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
import math
import enum
import torch.nn.init as init
import scipy.sparse as sp


class DNN(nn.Module):
	"""
	A deep neural network for the reverse diffusion preocess.
	"""
	def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
		super(DNN, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
		self.time_type = time_type
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		if self.time_type == "cat":
			in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
		else:
			raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
		out_dims_temp = self.out_dims
		
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
			for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
			for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
		
		self.drop = nn.Dropout(dropout)
		self.init_weights()
 
	def init_weights(self):
		for layer in self.in_layers + self.out_layers:
			init.xavier_normal_(layer.weight)
			init.normal_(layer.bias, mean=0.0, std=0.001)
	
	def forward(self, x, timesteps): # timestep应该是x_t的t
		time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)
		
		return h


def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.

	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""

	half = dim // 2
	freqs = torch.exp(
		-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
	).to(timesteps.device)
	args = timesteps[:, None].float() * freqs[None]
	embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2:
		embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
	return embedding

class ModelMeanType(enum.Enum):
	START_X = enum.auto()  # the model predicts x_0
	EPSILON = enum.auto()  # the model predicts epsilon

class GaussianDiffusion(nn.Module):
	def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
			steps, device, history_num_per_term=10, beta_fixed=True):

		self.mean_type = mean_type
		self.noise_schedule = noise_schedule
		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps
		self.device = device

		self.history_num_per_term = history_num_per_term
		self.Lt_history = torch.zeros(steps, history_num_per_term, dtype=torch.float64).to(device)
		self.Lt_count = torch.zeros(steps, dtype=int).to(device)

		if noise_scale != 0.:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
			if beta_fixed:
				self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
				# The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
			assert len(self.betas.shape) == 1, "betas must be 1-D"
			assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
			assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

			self.calculate_for_diffusion()

		super(GaussianDiffusion, self).__init__()
	
	def get_betas(self):
		"""
		Given the schedule name, create the betas for the diffusion process.
		"""
		if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
			start = self.noise_scale * self.noise_min
			end = self.noise_scale * self.noise_max
			if self.noise_schedule == "linear":
				return np.linspace(start, end, self.steps, dtype=np.float64)
			else:
				return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
		elif self.noise_schedule == "cosine":
			return betas_for_alpha_bar(
			self.steps,
			lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
		)
		elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
			ts = np.arange(self.steps)
			betas = [1 / (self.steps - t + 1) for t in ts]
			return betas
		else:
			raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
	
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
		assert self.alphas_cumprod_prev.shape == (self.steps,)

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)

		self.posterior_log_variance_clipped = torch.log(
			torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
		)
		self.posterior_mean_coef1 = (
			self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_mean_coef2 = (
			(1.0 - self.alphas_cumprod_prev)
			* torch.sqrt(alphas)
			/ (1.0 - self.alphas_cumprod)
		)
	
	def p_sample(self, model, x_start, steps, sampling_noise=False):
		assert steps <= self.steps, "Too much steps in inference."
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
			x_t = self.q_sample(x_start, t)

		indices = list(range(self.steps))[::-1]

		if self.noise_scale == 0.:
			for i in indices:
				t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
				x_t = model(x_t, t)
			return x_t

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
			out = self.p_mean_variance(model, x_t, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = (
					(t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
				)  # no noise when t == 0
				x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
			else:
				x_t = out["mean"]
		return x_t
	
	def training_losses(self, model, x_start, reweight=False):
		batch_size, device = x_start.size(0), x_start.device
		ts, pt = self.sample_timesteps(batch_size, device, 'importance')
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0.:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		terms = {}
		model_output = model(x_t, ts)
		target = {
			ModelMeanType.START_X: x_start,
			ModelMeanType.EPSILON: noise,
		}[self.mean_type]

		assert model_output.shape == target.shape == x_start.shape

		mse = mean_flat((target - model_output) ** 2)

		if reweight == True:
			if self.mean_type == ModelMeanType.START_X:
				weight = self.SNR(ts - 1) - self.SNR(ts)
				weight = torch.where((ts == 0), 1.0, weight)
				loss = mse
			elif self.mean_type == ModelMeanType.EPSILON:
				weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
				weight = torch.where((ts == 0), 1.0, weight)
				likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
				loss = torch.where((ts == 0), likelihood, mse)
		else:
			weight = torch.tensor([1.0] * len(target)).to(device)
			if self.mean_type == ModelMeanType.START_X:
				loss = mse
			elif self.mean_type == ModelMeanType.EPSILON:
				likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
				loss = torch.where((ts == 0), likelihood, mse)

		terms["loss"] = weight * loss
		
		# update Lt_history & Lt_count
		for t, loss in zip(ts, terms["loss"]):
			if self.Lt_count[t] == self.history_num_per_term:
				Lt_history_old = self.Lt_history.clone()
				self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
				self.Lt_history[t, -1] = loss.detach()
			else:
				try:
					self.Lt_history[t, self.Lt_count[t]] = loss.detach()
					self.Lt_count[t] += 1
				except:
					print(t)
					print(self.Lt_count[t])
					print(loss)
					raise ValueError

		terms["loss"] /= pt
		return terms

	def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
		if method == 'importance':  # importance sampling
			if not (self.Lt_count == self.history_num_per_term).all():
				return self.sample_timesteps(batch_size, device, method='uniform')
			
			Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
			pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
			pt_all *= 1- uniform_prob
			pt_all += uniform_prob / len(pt_all)

			assert pt_all.sum(-1) - 1. < 1e-5

			t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
			pt = pt_all.gather(dim=0, index=t) * len(pt_all)

			return t, pt
		
		elif method == 'uniform':  # uniform sampling
			t = torch.randint(0, self.steps, (batch_size,), device=device).long()
			pt = torch.ones_like(t).float()

			return t, pt
			
		else:
			raise ValueError
	
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		assert noise.shape == x_start.shape
		return (
			self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
			+ self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
			* noise
		)
	
	def q_posterior_mean_variance(self, x_start, x_t, t):
		"""
		Compute the mean and variance of the diffusion posterior:
			q(x_{t-1} | x_t, x_0)
		"""
		assert x_start.shape == x_t.shape
		posterior_mean = (
			self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
			+ self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
		)
		posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
		posterior_log_variance_clipped = self._extract_into_tensor(
			self.posterior_log_variance_clipped, t, x_t.shape
		)
		assert (
			posterior_mean.shape[0]
			== posterior_variance.shape[0]
			== posterior_log_variance_clipped.shape[0]
			== x_start.shape[0]
		)
		return posterior_mean, posterior_variance, posterior_log_variance_clipped
	
	def p_mean_variance(self, model, x, t):
		"""
		Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
		the initial x, x_0.
		"""
		B, C = x.shape[:2]
		assert t.shape == (B, )
		model_output = model(x, t)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
		
		if self.mean_type == ModelMeanType.START_X:
			pred_xstart = model_output
		elif self.mean_type == ModelMeanType.EPSILON:
			pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
		else:
			raise NotImplementedError(self.mean_type)
		
		model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

		assert (
			model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
		)

		return {
			"mean": model_mean,
			"variance": model_variance,
			"log_variance": model_log_variance,
			"pred_xstart": pred_xstart,
		}

	
	def _predict_xstart_from_eps(self, x_t, t, eps):
		assert x_t.shape == eps.shape
		return (
			self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
			- self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
		)
	
	def SNR(self, t):
		"""
		Compute the signal-to-noise ratio for a single timestep.
		"""
		self.alphas_cumprod = self.alphas_cumprod.to(t.device)
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
	
	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		"""
		Extract values from a 1-D numpy array for a batch of indices.

		:param arr: the 1-D numpy array.
		:param timesteps: a tensor of indices into the array to extract.
		:param broadcast_shape: a larger shape of K dimensions with the batch
								dimension equal to the length of timesteps.
		:return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
		"""
		# res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
		arr = arr.to(timesteps.device)
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

def betas_from_linear_variance(steps, variance, max_beta=0.999):
	alpha_bar = 1 - variance
	betas = []
	betas.append(1 - alpha_bar[0])
	for i in range(1, steps):
		betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
	return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
	"""
	Create a beta schedule that discretizes the given alpha_t_bar function,
	which defines the cumulative product of (1-beta) over time from t = [0,1].

	:param num_diffusion_timesteps: the number of betas to produce.
	:param alpha_bar: a lambda that takes an argument t from 0 to 1 and
					  produces the cumulative product of (1-beta) up to that
					  part of the diffusion process.
	:param max_beta: the maximum beta to use; use values lower than 1 to
					 prevent singularities.
	"""
	betas = []
	for i in range(num_diffusion_timesteps):
		t1 = i / num_diffusion_timesteps
		t2 = (i + 1) / num_diffusion_timesteps
		betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
	return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
	"""
	Compute the KL divergence between two gaussians.

	Shapes are automatically broadcasted, so batches can be compared to
	scalars, among other use cases.
	"""
	tensor = None
	for obj in (mean1, logvar1, mean2, logvar2):
		if isinstance(obj, torch.Tensor):
			tensor = obj
			break
	assert tensor is not None, "at least one argument must be a Tensor"

	# Force variances to be Tensors. Broadcasting helps convert scalars to
	# Tensors, but it does not work for torch.exp().
	logvar1, logvar2 = [
		x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
		for x in (logvar1, logvar2)
	]

	return 0.5 * (
		-1.0
		+ logvar2
		- logvar1
		+ torch.exp(logvar1 - logvar2)
		+ ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
	)

def mean_flat(tensor):
	"""
	Take the mean over all non-batch dimensions.
	"""
	return tensor.mean(dim=list(range(1, len(tensor.shape))))



class DiffRec(GeneralModel):
	reader, runner = 'myReader', 'myRunner'
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
		# DNN的参数
		parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
		parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
		parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
		parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
		# GaussianDiffusion的参数
		parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
		parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
		parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
		parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
		parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
		parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
		parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
		parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
		parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.time_type = args.time_type
		self.emb_size = args.emb_size
		self.norm = args.norm
		self.dims = eval(args.dims)
		# GaussianDiffusion的参数
		self.mean_type = args.mean_type
		self.steps = args.steps
		self.noise_schedule = args.noise_schedule
		self.noise_scale = args.noise_scale
		self.noise_min = args.noise_min
		self.noise_max = args.noise_max
		self.sampling_noise = args.sampling_noise
		self.sampling_steps = args.sampling_steps
		self.reweight = args.reweight

		
		# 将DNN和GaussianDiffusion初始化
		### Build Gaussian Diffusion ###
		if args.mean_type == 'x0':
			mean_type = ModelMeanType.START_X
		elif args.mean_type == 'eps':
			mean_type = ModelMeanType.EPSILON
		else:
			raise ValueError("Unimplemented mean type %s" % args.mean_type)

		out_dims = self.dims + [self.item_num]
		in_dims = out_dims[::-1]
		self.DNN = DNN(in_dims, out_dims, self.emb_size, time_type="cat", norm=self.norm).to(self.device)
		self.GaussianDiffusion = GaussianDiffusion(mean_type, self.noise_schedule, self.noise_scale, self.noise_min, self.noise_max, self.steps, self.device).to(self.device)
		
		param_num = 0
		mlp_num = sum([param.nelement() for param in self.DNN.parameters()])
		diff_num = sum([param.nelement() for param in self.GaussianDiffusion.parameters()])
		param_num = mlp_num + diff_num
		print("Number of all parameters:", param_num)

	def forward(self, batch_data: torch.Tensor):
		"""
		:param batch_data: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		# generate prediction (ranking score according to tensors in feed_dict)
		
		# 使用DNN
		prediction = self.GaussianDiffusion.p_sample(self.DNN, batch_data, self.sampling_steps, self.sampling_noise)
		
		out_dict = {'prediction': prediction}
		return out_dict

	def loss(self, batch_data) -> torch.Tensor:
		losses = self.GaussianDiffusion.training_losses(self.DNN, batch_data, self.reweight)  # 将模型、batch输入和是否给不同时间步分配不同的权重，默认超参数是true
		loss = losses["loss"].mean()  # loss这里是取平均值!
		return loss


	class Dataset(GeneralModel.Dataset):
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict() # 如果设定了要缓存则除了训练集，其他集合的数据都会缓存，将通过_get_feed_dict得到的数据进行缓存
			self.data = corpus.data_df[phase].to_dict('list') #是字典
			# ↑ DataFrame is not compatible with multi-thread operations

			self.csr_matrix = sp.csr_matrix((np.ones_like(corpus.data_df[phase]['user_id']), (corpus.data_df[phase]['user_id'], corpus.data_df[phase]['item_id'])),
								   dtype='float32', shape=(self.corpus.n_users, self.corpus.n_items))
			self.csr_matrix_A = torch.FloatTensor(self.csr_matrix.A)

		def __len__(self):
			return len(self.csr_matrix_A)

		# ! Key method to construct input data for a single instance
		def __getitem__(self, index: int) -> dict:
			'''
			返回的是交互矩阵的一行
   			'''
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			return self.csr_matrix_A[index]

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self.csr_matrix_A[i]

		def actions_before_epoch(self):
			pass

