"""GPN-Star + ChromBPNet, mirroring ArsenalChromBPNet with batch['msa'] from GPNStarDataset."""

import numpy as np
import torch
import torch.nn as nn

from gpn import register_auto_classes
from gpn.data import ReverseComplementer, Tokenizer, slice_alignment
from gpn.star.data import GenomeMSA
from gpn.star.utils import find_directory_sum_paths
from transformers import AutoModel

from .bpnet import BPNet
from .chrombpnet import _Exp, _Log


def chunk_layout(seq_len, chunk_size):
	"""ARSENAL's center-out tiling as (window_start, keep_start, keep_end) per chunk."""
	if chunk_size >= seq_len:
		return [(0, 0, seq_len)]
	first_start = seq_len // 2 - chunk_size // 2
	layout = [(first_start, first_start, first_start + chunk_size)]

	right_start = first_start + chunk_size
	while right_start < seq_len:
		if right_start + chunk_size <= seq_len:
			layout.append((right_start, right_start, right_start + chunk_size))
			right_start += chunk_size
		else:
			layout.append((seq_len - chunk_size, right_start, seq_len))
			break

	left_end = first_start
	while left_end > 0:
		if left_end - chunk_size >= 0:
			layout.append((left_end - chunk_size, left_end - chunk_size, left_end))
			left_end -= chunk_size
		else:
			layout.append((0, 0, left_end))
			break
	return layout


class GPNStarEmbedder(nn.Module):
	"""Per-position GPN-Star embeddings for windows longer than its context."""

	def __init__(
		self,
		model_path,
		model_revision=None,
		chunk_size=256,
		num_layers_avg=4,
		chunk_batch_size=256,
	):
		super().__init__()
		register_auto_classes()
		self.model = AutoModel.from_pretrained(model_path, revision=model_revision)
		self.chunk_size = chunk_size
		self.num_layers_avg = num_layers_avg
		self.chunk_batch_size = chunk_batch_size
		n_layers = self.model.config.num_hidden_layers
		if not 1 <= num_layers_avg <= n_layers:
			raise ValueError(f"num_layers_avg must be in [1, {n_layers}], got {num_layers_avg}")

	@property
	def hidden_size(self):
		return self.model.config.hidden_size

	def embed_chunks(self, msa):
		"""(n, chunk, species) tokens -> (n, chunk, hidden) mean of the last num_layers_avg layers."""
		msa = msa.long()
		out = self.model(
			input_ids=msa[:, :, :1],
			source_ids=msa,
			target_species=torch.zeros(msa.shape[0], 1, dtype=torch.long, device=msa.device),
			output_hidden_states=True,
		)
		# hidden_states[0] is the embedding; [k] is block k's output.
		layers = out.hidden_states[-self.num_layers_avg:]
		return (sum(layers) / len(layers))[:, :, 0]

	def forward(self, msa):
		"""(batch, seq_len, species) tokens -> (batch, seq_len, hidden)."""
		batch, seq_len, _ = msa.shape
		layout = chunk_layout(seq_len, self.chunk_size)
		width = min(self.chunk_size, seq_len)
		# Chunk-major: layout entry k is rows [k * batch, (k + 1) * batch).
		chunks = torch.cat([msa[:, w:w + width] for w, _, _ in layout], dim=0)
		embs = torch.cat([
			self.embed_chunks(chunks[i:i + self.chunk_batch_size])
			for i in range(0, chunks.shape[0], self.chunk_batch_size)
		], dim=0)

		out = embs.new_empty(batch, seq_len, embs.shape[-1])
		for k, (w, a, b) in enumerate(layout):
			out[:, a:b] = embs[k * batch:(k + 1) * batch, a - w:b - w]
		return out


class GPNStarChromBPNet(nn.Module):
	"""ChromBPNet whose accessibility model reads GPN-Star embeddings."""

	def __init__(self, config, **kwargs):
		super().__init__()

		self.finetune = config.finetune_gpnstar
		self.gpnstar_model = GPNStarEmbedder(
			config.gpnstar_model,
			model_revision=config.gpnstar_revision,
			chunk_size=config.gpnstar_input_size,
			num_layers_avg=config.num_layers_avg,
			chunk_batch_size=config.gpnstar_chunk_batch_size,
		)
		if config.input_embedding_dim != self.gpnstar_model.hidden_size:
			raise ValueError(
				f"input_embedding_dim={config.input_embedding_dim} but "
				f"{config.gpnstar_model} has hidden_size={self.gpnstar_model.hidden_size}"
			)
		if not self.finetune:
			for param in self.gpnstar_model.parameters():
				param.requires_grad = False
			self.gpnstar_model.eval()

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
		print("Reinitializing with TF strategy")
		for m in self.model.modules():
			if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
				if hasattr(m, 'weight') and m.weight is not None:
					nn.init.xavier_uniform_(m.weight)
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.zeros_(m.bias)

	def get_embeddings(self, msa):
		if msa is None:
			raise ValueError("GPN-Star models need batch['msa']; use GPNStarDataset")
		if msa.shape[1] != self.seq_input_size:
			raise ValueError(f"msa has {msa.shape[1]} positions, expected {self.seq_input_size}")
		if self.finetune:
			return self.gpnstar_model(msa)
		with torch.no_grad():
			return self.gpnstar_model(msa)

	def forward(self, x, msa=None, **kwargs):
		"""x: one-hot for the bias model; msa: (batch, input_len, species) for the same window."""
		# BPNet transposes (batch, length, hidden) itself when dim 1 is not 4.
		x_embs = self.get_embeddings(msa)

		acc_profile, acc_counts = self.model(x_embs)
		bias_profile, bias_counts = self.bias(x)

		y_profile = acc_profile + bias_profile
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))

		# DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
		return y_profile.squeeze(1), y_counts


class GPNStarChromBPNetNoBias(GPNStarChromBPNet):
	def forward(self, x, msa=None, **kwargs):
		x_embs = self.get_embeddings(msa)
		acc_profile, acc_counts = self.model(x_embs)
		return acc_profile.squeeze(1), acc_counts


class MSAReader:
	"""Tokenized windows from a gpn MSA directory; stores open lazily per DataLoader worker."""

	def __init__(self, msa_path, chroms=None):
		self.msa_paths = find_directory_sum_paths(msa_path)
		self.n_species = sum(self.msa_paths)
		self.chroms = None if chroms is None else [self.store_chrom(c) for c in chroms]
		self._stores = None

	@staticmethod
	def store_chrom(chrom):
		# Stores name chromosomes without "chr".
		return chrom[3:] if chrom.startswith("chr") else chrom

	def __getstate__(self):
		state = self.__dict__.copy()
		state["_stores"] = None
		return state

	@property
	def stores(self):
		if self._stores is None:
			self._stores = [
				GenomeMSA(path, n_species=n, subset_chroms=self.chroms, in_memory=False)
				for n, path in self.msa_paths.items()
			]
		return self._stores

	def get(self, chrom, start, end, strand):
		"""Same as GenomeMSA.get_msa(tokenize=True) but ~9x faster via a byte lookup table."""
		chrom = self.store_chrom(chrom)
		msa = np.concatenate(
			[slice_alignment(s.data[chrom], start, end, s.n_species).view(np.uint8) for s in self.stores],
			axis=-1,
		)
		if strand == "-":
			return _RC_TOKEN_TABLE[msa[::-1]]
		return _TOKEN_TABLE[msa]


def _token_tables():
	tokenizer, rc = Tokenizer(), ReverseComplementer()
	all_bytes = np.arange(256, dtype=np.uint8).view("S1")
	return tokenizer(all_bytes), tokenizer(rc.table)


_TOKEN_TABLE, _RC_TOKEN_TABLE = _token_tables()
