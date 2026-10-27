"""Gradient importance logos (no-bias log counts) at example K562 test loci for ChromBPNet, phyloP channel and GPN-Star."""

import argparse
import os
import sys
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyBigWig
import pyfaidx
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

sys.path.insert(0, '/pollard/home/szhang20/arsenal-chrombpnet/analysis/plots')
sys.path.insert(0, '/pollard/home/szhang20/arsenal-chrombpnet')
from plot_tracks import G, OBSERVED, PHYLOP, CONSERVED, pick_loci, pick_vs_arsenal
from chrombpnet.data_utils import dna_to_one_hot
from chrombpnet.gpnstar import MSAReader, chunk_layout
from chrombpnet.model_wrappers import ArsenalChromBPNetWrapper, load_pretrained_model

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']

OUT = '/pollard/home/szhang20/arsenal-chrombpnet/figs/gradients'
DATE = date.today().strftime('%Y%m%d')
FASTA = '/pollard/home/szhang20/.cache/genome/hg38/hg38.fa'
INPUT_LEN, FLANK = 2114, 1057
CKPT = 'fold_0/checkpoints/best_model.ckpt'
# Label, model type, run dir.
MODELS = (
	('ChromBPNet (paper)', 'chrombpnet', f'{G}/arsenal_download/ChromBPNet_Models/K562/run_1'),
	('ARSENAL (paper)', 'arsenal-chrombpnet', f'{G}/arsenal_download/ARSENAL_ChromBPNet_Models/K562/run_1'),
	('GPN-Star m447 25m last6', 'gpnstar-chrombpnet', f'{G}/chrombpnet_layers/runs/gpnstar_25m_last6'),
	('GPN-Star v100 25m last6', 'gpnstar-chrombpnet', f'{G}/chrombpnet_layers/runs/gpnstar_v100_25m_last6'),
)
REFERENCE = MODELS[0][0]
# ARSENAL LM rebuilt from ~/arsenal code (8 RoPE layers, 8 heads, post-norm); reproduces the authors' test pred_count.
ARSENAL_LM = '/pollard/data/projects/shzhang/gpn/arsenal_rebuilt/arsenal_lm_8layers_h8_normlast_untrained.pth'
SEG = 250
# Letters below this fraction of a model's largest |score| are left out.
MIN_FRAC = 0.15
COLORS = {'A': '#109648', 'C': '#255c99', 'G': '#f7b32b', 'T': '#d62839'}


def load(model_type, run):
	if model_type == 'arsenal-chrombpnet':
		# The authors' LM .pth is not here, so pass a rebuilt architecture whose weights the checkpoint then fills.
		hp = torch.load(f'{run}/{CKPT}', map_location='cpu', weights_only=False)['hyper_parameters']['args']
		hp.arsenal_model = ARSENAL_LM
		w = ArsenalChromBPNetWrapper.load_from_checkpoint(f'{run}/{CKPT}', map_location='cpu', weights_only=False, args=hp)
	else:
		w = load_pretrained_model(argparse.Namespace(model_type=model_type, bias_scaled=None, data_dir=''), f'{run}/{CKPT}')
	net = w.model.eval()
	net.hparams_msa = getattr(getattr(w.hparams, 'args', None), 'msa_path', None)
	return net


def onehot_grad(net, x):
	"""Mean-centered gradient of no-bias log counts w.r.t. the (1, 4 or 5, L) input, first 4 rows only."""
	x = torch.tensor(x, requires_grad=True)
	# BPNet transposes inputs whose dim 1 is not 4, so phyloP inputs go in as (1, L, 5).
	_, counts = net.model(x if x.shape[1] == 4 else x.transpose(1, 2))
	counts.sum().backward()
	g = x.grad[0, :4].numpy()
	return g - g.mean(axis=0, keepdims=True)


def arsenal_windows(seq_len, chunk):
	"""Window starts in the order ArsenalChromBPNet.get_embeddings embeds them: center, rightward, then leftward."""
	first = seq_len // 2 - chunk // 2
	starts, right = [first], first + chunk
	while right < seq_len:
		starts.append(right if right + chunk <= seq_len else seq_len - chunk)
		right += chunk
	left = first
	while left > 0:
		starts.append(left - chunk if left - chunk >= 0 else 0)
		left -= chunk
	return starts


def embedding_grad(layer, held, starts, width, rows):
	"""Per-base scores g . (E[b] - mean E) from token-embedding gradients, summed where chunks overlap."""
	E = layer.weight[rows].detach()
	E = E - E.mean(axis=0, keepdims=True)
	g = torch.cat([o.grad for o in held], dim=0)
	g = g[:, :, 0] if g.dim() == 4 else g
	out = np.zeros((4, INPUT_LEN), dtype=np.float32)
	for k, w in enumerate(starts):
		out[:, w:w + width] += (g[k] @ E.T).T.numpy()
	return out


def hooked_backward(layer, run):
	"""Backprop predicted log counts to the outputs of an embedding layer; returns those outputs with .grad set."""
	held = []

	def hook(_, __, out):
		out = out.detach().requires_grad_(True)
		held.append(out)
		return out

	h = layer.register_forward_hook(hook)
	try:
		run().sum().backward()
	finally:
		h.remove()
	return held


def arsenal_grad(net, x):
	"""ARSENAL gradient via its token embedding (tokens 0-3 are A, C, G, T)."""
	layer = net.arsenal_model.input_embedder.vocab_emb
	tokens = net.one_hot_to_tokens(torch.tensor(x).transpose(1, 2))
	held = hooked_backward(layer, lambda: net.model(net.get_embeddings(tokens))[1])
	return embedding_grad(layer, held, arsenal_windows(INPUT_LEN, net.arsenal_input_size), net.arsenal_input_size, slice(0, 4))


def gpnstar_grad(net, msa):
	"""Per-base gradient via the human-row token embedding: g . (E[b] - mean E) for b in ACGT."""
	emb = net.gpnstar_model
	layer = emb.model.target_embedding.input_embed
	held = hooked_backward(layer, lambda: net.model(emb(torch.as_tensor(msa[None])))[1])
	# Tokens 1-4 are A, C, G, T; the human clade is masked in column attention, so this row is the only human input.
	starts = [w for w, _, _ in chunk_layout(INPUT_LEN, emb.chunk_size)]
	return embedding_grad(layer, held, starts, min(emb.chunk_size, INPUT_LEN), slice(1, 5))


def letter(ax, ch, x, y, h):
	tp = TextPath((0, 0), ch, size=1, prop=FontProperties(family='DejaVu Sans', weight='bold'))
	bb = tp.get_extents()
	t = Affine2D().translate(-bb.x0, -bb.y0).scale(0.95 / bb.width, h / bb.height).translate(x + 0.025, y)
	ax.add_patch(PathPatch(t.transform_path(tp), color=COLORS[ch], lw=0))


def logo(ax, proj, seq, x0, lim):
	"""Projected scores as one letter per position above MIN_FRAC of the max; negative scores hang below zero."""
	for i, (v, ch) in enumerate(zip(proj, seq)):
		if ch in COLORS and abs(v) >= MIN_FRAC * lim / 1.1:
			letter(ax, ch, x0 + i, 0, v)
	ax.set_ylim(-lim, lim)
	ax.axhline(0, color='black', lw=0.4)


def plot(name, why, chrom, center, seq, attrs, obs, phylop):
	"""Observed 1 kb on top, then the whole 1 kb in SEG bp blocks of phyloP plus one logo per model."""
	out_start = center - 500
	# 1 kb output window within the 2114 bp input.
	o = slice(FLANK - 500, FLANK + 500)
	proj = {k: (a * dna_to_one_hot([seq])[0].T).sum(0)[o] for k, a in attrs.items()}
	out_seq = seq[o]
	# One y-scale per model across all blocks, so letter heights compare along the window.
	lims = {k: np.abs(p).max() * 1.1 or 1 for k, p in proj.items()}
	n_seg, per = 1000 // SEG, 1 + len(proj)
	ratios = [1.2] + ([0.8] + [2] * len(proj) + [0.5]) * n_seg
	fig, axes = plt.subplots(len(ratios), 1, figsize=(24, 0.8 * sum(ratios)), gridspec_kw={'height_ratios': ratios})
	axes[0].bar(np.arange(out_start, out_start + 1000), obs, width=1, color='black')
	for k in range(1, n_seg):
		axes[0].axvline(out_start + k * SEG, color='#999999', ls=':', lw=0.8)
	axes[0].set_xlim(out_start, out_start + 1000)
	axes[0].set_ylabel('observed', rotation=0, ha='right', va='center', fontsize=9)
	axes[0].ticklabel_format(axis='x', style='plain', useOffset=False)
	for k in range(n_seg):
		a, b = k * SEG, (k + 1) * SEG
		x0 = out_start + a
		block = axes[1 + k * (per + 1):1 + (k + 1) * (per + 1)]
		ph = phylop[a:b]
		ax = block[0]
		ax.bar(np.arange(x0, x0 + SEG) + 0.5, np.clip(np.nan_to_num(ph), -4, None), width=0.9, color=np.where(ph > CONSERVED, '#c0392b', '#bbbbbb'))
		# Unaligned bases (no phyloP) as a black strip along the bottom.
		ax.fill_between(np.arange(x0, x0 + SEG) + 0.5, -4, -3, where=np.isnan(ph), color='black', lw=0, step='mid')
		ax.axhline(CONSERVED, color='black', ls=':', lw=0.6)
		ax.set_ylim(-4, 8)
		ax.set_ylabel('phyloP', rotation=0, ha='right', va='center', fontsize=8)
		for ax, (label, pr) in zip(block[1:per], proj.items()):
			logo(ax, pr[a:b], out_seq[a:b], x0, lims[label])
			ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8)
			if k == 0:
				r = np.corrcoef(pr, proj[REFERENCE])[0, 1]
				ax.text(1.005, 0.5, f'r vs ChromBPNet\n(1 kb) {r:.2f}', transform=ax.transAxes, va='center', fontsize=7)
		for ax in block[:per]:
			ax.set_xlim(x0, x0 + SEG)
			ax.ticklabel_format(axis='x', style='plain', useOffset=False)
			ax.spines[['top', 'right']].set_visible(False)
			ax.tick_params(labelsize=7)
		for ax in block[:per - 1]:
			ax.set_xticklabels([])
		block[-1].axis('off')
	axes[0].spines[['top', 'right']].set_visible(False)
	axes[0].tick_params(labelsize=7)
	fig.suptitle(f'{chrom}:{out_start:,}-{out_start + 1000:,}  {why}\nWhole 1 kb output window in {n_seg} blocks of {SEG} bp (dotted lines on top)', fontsize=11, x=0.02, ha='left')
	fig.text(0.02, 0.0, 'Letter height = gradient of the no-bias model\'s predicted log counts w.r.t. each base (mean-centered over A/C/G/T), shown for the base present when it is at least 15% of that model\'s largest score; below zero means the base lowers predicted accessibility. ARSENAL and GPN-Star gradients are taken at their (human-row) token embeddings. Each model keeps one y-scale across blocks. phyloP: red above 2, black strip = not in the alignment.', fontsize=8, va='top', wrap=True)
	fig.tight_layout()
	fig.savefig(f'{OUT}/{DATE}_K562_{name}.png', dpi=150, bbox_inches='tight')
	plt.close(fig)


def main():
	os.makedirs(OUT, exist_ok=True)
	torch.set_num_threads(20)
	nets = {label: (mt, load(mt, run)) for label, mt, run in MODELS}
	# Each GPN-Star run reads the alignment it was trained on.
	readers = {label: MSAReader(net.hparams_msa) for label, (mt, net) in nets.items() if mt == 'gpnstar-chrombpnet'}
	fasta, obs_bw, phylop_bw = pyfaidx.Fasta(FASTA), pyBigWig.open(OBSERVED), pyBigWig.open(PHYLOP)
	loci = [(f'locus{i}', why, r) for i, (why, r) in enumerate(pick_loci(), 1)]
	loci += [(f'gpn_beats_arsenal{i}', why, r) for i, (why, r) in enumerate(pick_vs_arsenal(), 1)]
	for name, why, row in loci:
		chrom, center = row['chr'], int(row['center'])
		start = center - FLANK
		seq = fasta[chrom][start:start + INPUT_LEN].seq.upper()
		x = dna_to_one_hot([seq]).transpose(0, 2, 1).astype(np.float32)
		phylop = np.array(phylop_bw.values(chrom, start, start + INPUT_LEN), dtype=np.float32)
		attrs = {}
		for label, (mt, net) in nets.items():
			if mt == 'chrombpnet':
				attrs[label] = onehot_grad(net, x)
			elif mt == 'arsenal-chrombpnet':
				attrs[label] = arsenal_grad(net, x)
			else:
				attrs[label] = gpnstar_grad(net, readers[label].get(chrom, start, start + INPUT_LEN, '+'))
		obs = np.nan_to_num(np.array(obs_bw.values(chrom, center - 500, center + 500)))
		# Unaligned bases are stored as 0 in this bigwig.
		ph = phylop[FLANK - 500:FLANK + 500].copy()
		ph[ph == 0] = np.nan
		plot(name, why, chrom, center, seq, attrs, obs, ph)
		print(f'{name}: {chrom}:{center} done', flush=True)
	print(f'plots in {OUT}')


if __name__ == '__main__':
	main()
