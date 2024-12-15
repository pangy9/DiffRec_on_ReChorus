# -*- coding: UTF-8 -*-
from models.BaseModel import GeneralModel
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_
from kmeans_pytorch import kmeans
import enum
from my_new_models.DiffRec import DiffRec
from models.BaseModel import GeneralModel

class AutoEncoder(nn.Module):
	"""
	Guassian Diffusion for large-scale recommendation.
	"""
	def __init__(self, item_emb, n_cate, in_dims, out_dims, device, act_func, reparam=True, dropout=0.1):
		super(AutoEncoder, self).__init__()

		self.item_emb = item_emb
		self.n_cate = n_cate
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.act_func = act_func
		self.n_item = len(item_emb)
		self.reparam = reparam
		self.dropout = nn.Dropout(dropout)

		if n_cate == 1:  # no clustering
			in_dims_temp = [self.n_item] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
			out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item]

			encoder_modules = []
			for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
				encoder_modules.append(nn.Linear(d_in, d_out))
				if self.act_func == 'relu':
					encoder_modules.append(nn.ReLU())
				elif self.act_func == 'sigmoid':
					encoder_modules.append(nn.Sigmoid())
				elif self.act_func == 'tanh':
					encoder_modules.append(nn.Tanh())
				else:
					raise ValueError
			self.encoder = nn.Sequential(*encoder_modules)

			decoder_modules = []
			for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
				decoder_modules.append(nn.Linear(d_in, d_out))
				if self.act_func == 'relu':
					decoder_modules.append(nn.ReLU())
				elif self.act_func == 'sigmoid':
					decoder_modules.append(nn.Sigmoid())
				elif self.act_func == 'tanh':
					decoder_modules.append(nn.Tanh())
				elif self.act_func == 'leaky_relu':
					encoder_modules.append(nn.LeakyReLU())
				else:
					raise ValueError
			decoder_modules.pop()
			self.decoder = nn.Sequential(*decoder_modules)
		
		else:
			self.cluster_ids, _ = kmeans(X=item_emb, num_clusters=n_cate, distance='euclidean', device=device)
			# cluster_ids(labels): [0, 1, 2, 2, 1, 0, 0, ...]
			category_idx = []
			for i in range(n_cate):
				idx = np.argwhere(self.cluster_ids.numpy() == i).squeeze().tolist()
				category_idx.append(torch.tensor(idx, dtype=int))
			self.category_idx = category_idx  # [cate1: [iid1, iid2, ...], cate2: [iid3, iid4, ...], cate3: [iid5, iid6, ...]]
			self.category_map = torch.cat(tuple(category_idx), dim=-1)  # map
			self.category_len = [len(self.category_idx[i]) for i in range(n_cate)]  # item num in each category
			print("category length: ", self.category_len)
			assert sum(self.category_len) == self.n_item

			##### Build the Encoder and Decoder #####
			encoder_modules = [[] for _ in range(n_cate)]
			decode_dim = []
			for i in range(n_cate):
				if i == n_cate - 1:
					latent_dims = list(self.in_dims - np.array(decode_dim).sum(axis=0))
				else:
					latent_dims = [int(self.category_len[i] / self.n_item * self.in_dims[j]) for j in range(len(self.in_dims))]
					latent_dims = [latent_dims[j] if latent_dims[j] != 0 else 1 for j in range(len(self.in_dims))]
				in_dims_temp = [self.category_len[i]] + latent_dims[:-1] + [latent_dims[-1] * 2]
				decode_dim.append(latent_dims)
				for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
					encoder_modules[i].append(nn.Linear(d_in, d_out))
					if self.act_func == 'relu':
						encoder_modules[i].append(nn.ReLU())
					elif self.act_func == 'sigmoid':
						encoder_modules[i].append(nn.Sigmoid())
					elif self.act_func == 'tanh':
						encoder_modules[i].append(nn.Tanh())
					elif self.act_func == 'leaky_relu':
						encoder_modules[i].append(nn.LeakyReLU())
					else:
						raise ValueError

			self.encoder = nn.ModuleList([nn.Sequential(*encoder_modules[i]) for i in range(n_cate)])
			print("Latent dims of each category: ", decode_dim)

			self.decode_dim = [decode_dim[i][::-1] for i in range(len(decode_dim))]

			if len(out_dims) == 0:  # one-layer decoder: [encoder_dim_sum, n_item]
				out_dim = self.in_dims[-1]
				decoder_modules = []
				decoder_modules.append(nn.Linear(out_dim, self.n_item))
				self.decoder = nn.Sequential(*decoder_modules)
			else:  # multi-layer decoder: [encoder_dim, hidden_size, cate_num]
				decoder_modules = [[] for _ in range(n_cate)]
				for i in range(n_cate):
					out_dims_temp = self.decode_dim[i] + [self.category_len[i]]
					for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
						decoder_modules[i].append(nn.Linear(d_in, d_out))
						if self.act_func == 'relu':
							decoder_modules[i].append(nn.ReLU())
						elif self.act_func == 'sigmoid':
							decoder_modules[i].append(nn.Sigmoid())
						elif self.act_func == 'tanh':
							decoder_modules[i].append(nn.Tanh())
						elif self.act_func == 'leaky_relu':
							encoder_modules[i].append(nn.LeakyReLU())
						else:
							raise ValueError
					decoder_modules[i].pop()
				self.decoder = nn.ModuleList([nn.Sequential(*decoder_modules[i]) for i in range(n_cate)])
			
		self.apply(xavier_normal_initialization)
		
	def Encode(self, batch):
		batch = self.dropout(batch)
		if self.n_cate == 1:
			hidden = self.encoder(batch)
			mu = hidden[:, :self.in_dims[-1]]
			logvar = hidden[:, self.in_dims[-1]:]

			if self.training and self.reparam:
				latent = self.reparamterization(mu, logvar)
			else:
				latent = mu
			
			kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

			return batch, latent, kl_divergence

		else: 
			batch_cate = []
			for i in range(self.n_cate):
				batch_cate.append(batch[:, self.category_idx[i]])
			# [batch_size, n_items] -> [[batch_size, n1_items], [batch_size, n2_items], [batch_size, n3_items]]
			latent_mu = []
			latent_logvar = []
			for i in range(self.n_cate):
				hidden = self.encoder[i](batch_cate[i])
				latent_mu.append(hidden[:, :self.decode_dim[i][0]])
				latent_logvar.append(hidden[:, self.decode_dim[i][0]:])
			# latent: [[batch_size, latent_size1], [batch_size, latent_size2], [batch_size, latent_size3]]

			mu = torch.cat(tuple(latent_mu), dim=-1)
			logvar = torch.cat(tuple(latent_logvar), dim=-1)
			if self.training and self.reparam:
				latent = self.reparamterization(mu, logvar)
			else:
				latent = mu

			kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

			return torch.cat(tuple(batch_cate), dim=-1), latent, kl_divergence
	
	def reparamterization(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)
	
	def Decode(self, batch):
		if len(self.out_dims) == 0 or self.n_cate == 1:  # one-layer decoder
			return self.decoder(batch)
		else:
			batch_cate = []
			start=0
			for i in range(self.n_cate):
				end = start + self.decode_dim[i][0]
				batch_cate.append(batch[:, start:end])
				start = end
			pred_cate = []
			for i in range(self.n_cate):
				pred_cate.append(self.decoder[i](batch_cate[i]))
			pred = torch.cat(tuple(pred_cate), dim=-1)

			return pred
	
def compute_loss(recon_x, x):
	return -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))  # multinomial log likelihood in MultVAE


def xavier_normal_initialization(module):
	r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
	nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
	using constant 0 to initialize.
	.. _`xavier_normal_`:
		https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
	Examples:
		>>> self.apply(xavier_normal_initialization)
	"""
	import torch.nn.init as init
	if isinstance(module, nn.Linear):
		init.xavier_normal_(module.weight)
		if module.bias is not None:
			init.constant_(module.bias, 0.0)
	# if isinstance(module, nn.Linear):
	# 	xavier_normal_(module.weight.data)
	# 	if module.bias is not None:
	# 		constant_(module.bias.data, 0)            


class DNN(nn.Module):
	"""
	A deep neural network for the reverse process of latent diffusion.
	"""
	def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, act_func='tanh', dropout=0.5):
		super(DNN, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
		self.time_emb_dim = emb_size
		self.time_type = time_type
		self.norm = norm
		
		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		if self.time_type == "cat":
			in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
		else:
			raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
		out_dims_temp = self.out_dims

		self.in_modules = []
		for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
			self.in_modules.append(nn.Linear(d_in, d_out))
			if act_func == 'tanh':
				self.in_modules.append(nn.Tanh())
			elif act_func == 'relu':
				self.in_modules.append(nn.ReLU())
			elif act_func == 'sigmoid':
				self.in_modules.append(nn.Sigmoid())
			elif act_func == 'leaky_relu':
				self.in_modules.append(nn.LeakyReLU())
			else:
				raise ValueError
		self.in_layers = nn.Sequential(*self.in_modules)

		self.out_modules = []
		for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]):
			self.out_modules.append(nn.Linear(d_in, d_out))
			if act_func == 'tanh':
				self.out_modules.append(nn.Tanh())
			elif act_func == 'relu':
				self.out_modules.append(nn.ReLU())
			elif act_func == 'sigmoid':
				self.out_modules.append(nn.Sigmoid())
			elif act_func == 'leaky_relu':
				self.out_modules.append(nn.LeakyReLU())
			else:
				raise ValueError
		self.out_modules.pop()
		self.out_layers = nn.Sequential(*self.out_modules)

		self.dropout = nn.Dropout(dropout)

		self.apply(xavier_normal_initialization)
	
	def forward(self, x, timesteps):
		time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x, dim=-1)
		x = self.dropout(x)
		h = torch.cat([x, emb], dim=-1)
		h = self.in_layers(h)
		h = self.out_layers(h)

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

def xavier_normal_initialization(module):
	r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
	nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
	using constant 0 to initialize.
	.. _`xavier_normal_`:
		https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
	Examples:
		>>> self.apply(xavier_normal_initialization)
	"""
	if isinstance(module, nn.Linear):
		xavier_normal_(module.weight.data)
		if module.bias is not None:
			constant_(module.bias.data, 0)         



class ModelMeanType(enum.Enum):
	START_X = enum.auto()  # the model predicts x_0
	EPSILON = enum.auto()  # the model predicts epsilon

class GaussianDiffusion(nn.Module):
	def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
			steps, device, history_num_per_term=10, beta_fixed=False):

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

		if self.mean_type == ModelMeanType.START_X:
			terms["pred_xstart"] = model_output
		else:
			terms["pred_xstart"] = self._predict_xstart_from_eps(x_t, ts, model_output)
		
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


# 原本代码里面，embedding是从预训练的LightGCN里面得到的，因为框架刚好支持LightGCN，所以这里直接参考LightGCN的embedding
class LDiffRec(DiffRec):
	reader, runner = 'myReader', 'myRunner' # 使用自定义的Runner
	extra_log_args = {
		"lr1": "lr1",
		"lr2": "lr2",
		"wd1": "wd1",
		"wd2": "wd2",
		"batch_size": "bs",
		"n_cate": "nc",
		"in_dims": "idims",
		"out_dims": "odims",
		"lamda": "lamda",
		"mlp_dims": "mlpdims",
		"emb_size": "embs",
		"steps": "st",
		"noise_scale": "ns",
		"noise_min": "nmin",
		"noise_max": "nmax",
		"sampling_steps": "sample_st",
		"reweight": "rw"
	}
	@staticmethod
	def parse_model_args(parser):
		# Autoencoder的参数
		parser.add_argument('--n_cate', type=int, default=3, help='category num of items')
		parser.add_argument('--in_dims', type=str, default='[300]', help='the dims for the encoder')
		parser.add_argument('--out_dims', type=str, default='[]', help='the hidden dims for the decoder')
		parser.add_argument('--act_func', type=str, default='tanh', help='activation function for autoencoder')
		parser.add_argument('--lamda', type=float, default=0.03, help='hyper-parameter of multinomial log-likelihood for AE: 0.01, 0.02, 0.03, 0.05')
		parser.add_argument('--anneal_cap', type=float, default=0.005)
		parser.add_argument('--anneal_steps', type=int, default=500)
		parser.add_argument('--vae_anneal_cap', type=float, default=0.3)
		parser.add_argument('--vae_anneal_steps', type=int, default=200)
		parser.add_argument('--reparam', type=bool, default=True, help="Autoencoder with variational inference or not")
		parser.add_argument('--emb_path', type=str, default='./datasets/')
		# MLP
		parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
		parser.add_argument('--mlp_dims', type=str, default='[300]', help='the dims for the DNN')
		parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
		parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')
		parser.add_argument('--mlp_act_func', type=str, default='tanh', help='the activation function for MLP')

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
		# super().__init__(args, corpus)
		GeneralModel.__init__(self, args, corpus)
		# Autoencoder的参数
		self.n_cate = args.n_cate
		self.in_dims = eval(args.in_dims)[::-1]  # 将字符串转换为列表
		self.out_dims = eval(args.out_dims)  # 将字符串转换为列表
		self.act_func = args.act_func
		self.lamda = args.lamda
		self.anneal_cap = args.anneal_cap
		self.anneal_steps = args.anneal_steps
		self.vae_anneal_cap = args.vae_anneal_cap
		self.vae_anneal_steps = args.vae_anneal_steps
		self.reparam = args.reparam
		self.optimizer1 = None

		# MLP的参数
		self.time_type = args.time_type
		self.mlp_dims = eval(args.mlp_dims)
		self.norm = args.norm
		self.emb_size = args.emb_size
		self.mlp_act_func = args.mlp_act_func
		self.optimizer1 = None

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
		
		# 记录模型的训练更新情况
		self.update_count_vae = 0
		self.update_count = 0
		
		item_emb = torch.from_numpy(np.load(args.emb_path, allow_pickle=True))
		print(f"item_emb shape: {item_emb.shape}")
		print(f"coprus n_items: {corpus.n_items}")
		assert len(item_emb) == corpus.n_items
		self.Autoencoder = AutoEncoder(item_emb, self.n_cate, self.in_dims, self.out_dims, self.device, self.act_func, self.reparam).to(self.device)


		
		# 将DNN和GaussianDiffusion初始化
		### Build Gaussian Diffusion ###
		if self.mean_type == 'x0':
			mean_type = ModelMeanType.START_X
		elif self.mean_type == 'eps':
			mean_type = ModelMeanType.EPSILON
		else:
			raise ValueError("Unimplemented mean type %s" % self.mean_type)

		latent_size = self.in_dims[-1]
		mlp_out_dims = self.mlp_dims + [latent_size]
		mlp_in_dims = mlp_out_dims[::-1]
		self.DNN = DNN(mlp_in_dims, mlp_out_dims, self.emb_size, time_type=self.time_type, norm=self.norm, act_func=self.mlp_act_func).to(self.device)
		self.GaussianDiffusion = GaussianDiffusion(mean_type, self.noise_schedule, self.noise_scale, self.noise_min, self.noise_max, self.steps, self.device).to(self.device)

  
	def forward(self, batch_data):
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		# generate prediction (ranking score according to tensors in feed_dict)
		
		# 使用DNN
		_, batch_latent, _ = self.Autoencoder.Encode(batch_data)
		batch_latent_recon = self.GaussianDiffusion.p_sample(self.DNN, batch_latent, self.sampling_steps, self.sampling_noise)
		prediction = self.Autoencoder.Decode(batch_latent_recon)  # [batch_size, n1_items + n2_items + n3_items]
		category_map = self.Autoencoder.category_map.to(self.device)
		reverse_map = {int(category_map[i]):i for i in range(len(category_map))} # 反向映射，原序:新序
		prediction = prediction[:, [reverse_map[i] for i in range(prediction.shape[1])]]# 按照原来的item_id排序
		
		out_dict = {'prediction': prediction}
		return out_dict

	def loss(self, batch_data) -> torch.Tensor:
		batch = batch_data.to(self.device)
		batch_cate, batch_latent, vae_kl = self.Autoencoder.Encode(batch)

		terms = self.GaussianDiffusion.training_losses(self.DNN, batch_latent, self.reweight)
		elbo = terms["loss"].mean()  # loss from diffusion
		batch_latent_recon = terms["pred_xstart"]

		batch_recon = self.Autoencoder.Decode(batch_latent_recon)

		if self.anneal_steps > 0:
			lamda = max((1. - self.update_count / self.anneal_steps) * self.lamda, self.anneal_cap)
		else:
			lamda = max(self.lamda, self.anneal_cap)
		
		if self.vae_anneal_steps > 0:
			anneal = min(self.vae_anneal_cap, 1. * self.update_count_vae / self.vae_anneal_steps)
		else:
			anneal = self.vae_anneal_cap

		vae_loss = compute_loss(batch_recon, batch_cate) + anneal * vae_kl  # loss from autoencoder
		
		if self.reweight:
			loss = lamda * elbo + vae_loss
		else:
			loss = elbo + lamda * vae_loss

		return loss

