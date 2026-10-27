"""Distilled GPN-Star (sequence-only) + ChromBPNet, with batch['tokens'] from DistillDataset."""

import numpy as np
import torch
import torch.nn as nn

from .bpnet import BPNet
from .chrombpnet import _Exp, _Log

# gpn_distill tokens: A=0, C=1, G=2, T=3, N=4; anything else reads as N.
# this is different than the original GPN? 
_TOKEN_TABLE = np.full(256, 4, dtype=np.int64)
for _b, _t in zip(b'ACGTacgt', (0, 1, 2, 3, 0, 1, 2, 3)):
	_TOKEN_TABLE[_b] = _t
# Complement of each token; N stays N.
_RC_TOKENS = np.array([3, 2, 1, 0, 4, 5], dtype=np.int64)


def tokenize(seq):
	return _TOKEN_TABLE[np.frombuffer(seq.encode(), dtype=np.uint8)]


def rc_tokens(tokens):
	return _RC_TOKENS[tokens[::-1]]


def fixed_rms_norm(h, eps=1e-5):
	"""Per-position RMSNorm with gain 1, in fp32 (gpn_distill's 'shared' normalization)."""
	h = h.float()
	return h * (h.square().mean(-1, keepdim=True) + eps).rsqrt()


class DistillEmbedder(nn.Module):
	"""Per-position distilled-model features for the central crop of a longer token window."""

	def __init__(self, model_dir, representation='shared', num_layers_avg=1, crop=2114, attn_impl='auto'):
		super().__init__()
		from gpn_distill import load_model
		self.model = load_model(model_dir, device='cpu')
		self.representation = representation
		self.num_layers_avg = num_layers_avg
		self.crop = crop
		self.attn_impl = attn_impl
		n_layers = len(self.model.layers)
		if not 1 <= num_layers_avg <= n_layers:
			raise ValueError(f"num_layers_avg must be in [1, {n_layers}], got {num_layers_avg}")
		heads = list(self.model.head_names)
		if representation not in ['shared'] + heads:
			raise ValueError(f"representation must be 'shared' or one of {heads}, got {representation}")
		# A head's learned gains belong to its final-layer norm, so they are not applied to averaged layers.
		if representation != 'shared' and num_layers_avg != 1:
			raise ValueError("head representations use the final layer only (num_layers_avg=1)")

	@property
	def hidden_size(self):
		return self.model.config.model_dim

	def head_gain(self):
		"""Learned RMSNorm gains of the chosen head, shape (hidden,)."""
		return self.model.distill_heads[self.model.head_names.index(self.representation)][0].weight

	def forward(self, tokens):
		"""(batch, context) tokens -> (batch, crop, hidden) fp32, centered on the window center."""
		start = (tokens.shape[1] - self.crop) // 2
		outs = []

		def hook(_, __, out):
			# Crop right away so long-context activations are not kept for every layer.
			outs.append(out[:, start:start + self.crop])

		hooks = [layer.register_forward_hook(hook) for layer in self.model.layers[-self.num_layers_avg:]]
		try:
			with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=tokens.is_cuda):
				h = self.model.embed_norm(self.model.embedding(tokens))
				for layer in self.model.layers:
					h = layer(h, attn_impl=self.attn_impl)
		finally:
			for hk in hooks:
				hk.remove()
		# Each layer is normalized on its own before averaging, so deeper (larger) layers do not dominate.
		emb = sum(fixed_rms_norm(o) for o in outs) / len(outs)
		if self.representation != 'shared':
			# Equals the head's own RMSNorm: x / rms(x) * g, eps 1e-5, here in fp32.
			emb = emb * self.head_gain().float()
		return emb


class DistillChromBPNet(nn.Module):
	"""ChromBPNet whose accessibility model reads frozen distilled-model features."""

	def __init__(self, config, **kwargs):
		super().__init__()
		self.distill_model = DistillEmbedder(
			config.distill_model,
			representation=config.distill_representation,
			num_layers_avg=config.num_layers_avg,
			crop=config.input_len,
			attn_impl=config.distill_attn,
		)
		if config.input_embedding_dim != self.distill_model.hidden_size:
			raise ValueError(f"input_embedding_dim={config.input_embedding_dim} but the distilled model has hidden_size={self.distill_model.hidden_size}")
		for param in self.distill_model.parameters():
			param.requires_grad = False
		self.distill_model.eval()

		self.seq_input_size = config.input_len
		self.model = BPNet(
			out_dim=config.out_dim,
			n_filters=config.n_filters,
			n_layers=config.n_layers,
			conv1_kernel_size=config.conv1_kernel_size,
			profile_kernel_size=config.profile_kernel_size,
			n_outputs=config.n_outputs,
			n_control_tracks=config.n_control_tracks,
			profile_output_bias=config.profile_output_bias,
			count_output_bias=config.count_output_bias,
		)
		# Embeddings replace the 4 one-hot channels at the first convolution.
		self.model.iconv = torch.nn.Conv1d(config.input_embedding_dim, config.n_filters, kernel_size=config.conv1_kernel_size, padding='valid')
		self.bias = BPNet(out_dim=config.out_dim, n_layers=4, n_filters=128)
		self._log = _Log()
		self._exp1 = _Exp()
		self._exp2 = _Exp()
		self.n_control_tracks = config.n_control_tracks
		self.tf_style_reinit()

	def tf_style_reinit(self):
		"""Xavier/Glorot uniform weights and zero biases for Linear and Conv1d, in place."""
		for m in self.model.modules():
			if isinstance(m, (nn.Conv1d, nn.Linear)):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def get_embeddings(self, tokens):
		if tokens is None:
			raise ValueError("Distilled models need batch['tokens']; use DistillDataset")
		with torch.no_grad():
			return self.distill_model(tokens)

	def forward(self, x, tokens=None, **kwargs):
		"""x: one-hot for the bias model; tokens: (batch, context) for the same window center."""
		acc_profile, acc_counts = self.model(self.get_embeddings(tokens))
		bias_profile, bias_counts = self.bias(x)
		y_profile = acc_profile + bias_profile
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))
		return y_profile.squeeze(1), y_counts


class DistillChromBPNetNoBias(DistillChromBPNet):
	def forward(self, x, tokens=None, **kwargs):
		acc_profile, acc_counts = self.model(self.get_embeddings(tokens))
		return acc_profile.squeeze(1), acc_counts
