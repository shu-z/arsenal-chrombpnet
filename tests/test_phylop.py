"""phyloP-channel ChromBPNet tests; run with ARSENAL_MODEL_DIR=~/arsenal pytest tests/test_phylop.py."""

from argparse import Namespace

import numpy as np
import pyBigWig
import pytest
import torch

from chrombpnet.dataset import PhyloPDataset, crop_revcomp_data
from chrombpnet.model_config import ChromBPNetConfig
from chrombpnet.phylop import PhyloPChromBPNet

ARGS = dict(
	alpha=1.0, beta=1.0, verbose=False,
	out_dim=1000, n_filters=16, n_layers=2, conv1_kernel_size=21, profile_kernel_size=75,
	n_outputs=1, n_control_tracks=0, profile_output_bias=True, count_output_bias=True, input_len=2114,
)


def _position_bigwig(path, length):
	"""Bigwig on chr1 whose value at each base is its zero-based position."""
	bw = pyBigWig.open(str(path), "w")
	bw.addHeader([("chr1", length)])
	bw.addEntries("chr1", 0, values=np.arange(length, dtype=np.float64).tolist(), span=1, step=1)
	bw.close()


def test_phylop_row_follows_crop_and_strand(tmp_path):
	"""Row 5 is phyloP at the same bases, in the same order, as rows 1-4 after jitter and revcomp."""
	rng = np.random.default_rng(0)
	inputlen, outputlen, jitter, length = 2114, 1000, 500, 20_000
	onehot = np.eye(4)[rng.integers(0, 4, length)]
	_position_bigwig(tmp_path / "pos.bw", length)

	def load(summits, width, out_width, is_peak):
		seqs = np.stack([onehot[s - width // 2:s + width // 2] for s in summits])
		return seqs, np.zeros((len(summits), out_width)), np.array([['chr1', s, 'f', is_peak] for s in summits])

	peaks = load(rng.integers(3000, 17000, 40), inputlen + 2 * jitter, outputlen + 2 * jitter, 1)
	nonpeaks = load(rng.integers(3000, 17000, 400), inputlen, outputlen, 0)
	seqs, cts, coords = crop_revcomp_data(*peaks, *nonpeaks, inputlen, outputlen,
		add_revcomp=True, negative_sampling_ratio=0.5)

	ds = object.__new__(PhyloPDataset)
	ds.inputlen, ds.cur_seqs, ds.cur_cts, ds.cur_coords = inputlen, seqs, cts, coords
	ds.phylop_bw, ds._bw = str(tmp_path / "pos.bw"), None
	assert {c[2] for c in coords} == {'f', 'r'} and {c[3] for c in coords} == {'0', '1'}
	for i in range(len(seqs)):
		x = ds[i]['onehot_seq']
		_, start, end, strand = ds.window(i)
		positions = np.arange(start, end)[::-1] if strand == '-' else np.arange(start, end)
		assert x.shape == (5, inputlen)
		assert np.array_equal(x[:4], seqs[i].T)
		assert np.array_equal(x[4], positions), (i, coords[i])


def test_missing_phylop_is_zero(tmp_path):
	bw = pyBigWig.open(str(tmp_path / "gap.bw"), "w")
	bw.addHeader([("chr1", 100)])
	bw.addEntries(["chr1"], [10], ends=[20], values=[3.0])
	bw.close()
	ds = object.__new__(PhyloPDataset)
	ds.phylop_bw, ds._bw = str(tmp_path / "gap.bw"), None
	vals = ds.phylop("chr1", 0, 30, "+")
	assert np.isfinite(vals).all() and vals[:10].sum() == 0 and (vals[10:20] == 3).all()


def test_bias_sees_dna_only_and_phylop_reaches_accessibility():
	model = PhyloPChromBPNet(ChromBPNetConfig(n_filters=16, n_layers=2)).eval()
	assert model.model.iconv.in_channels == 5
	seen = []
	model.bias.register_forward_hook(lambda m, inp, out: seen.append(inp[0].shape))

	x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 2114)), 4).float().transpose(1, 2)
	x = torch.cat([x, torch.zeros(2, 1, 2114)], dim=1)
	with torch.no_grad():
		profile, counts = model(x)
		x2 = x.clone()
		x2[:, 4] = 5.0
		profile2, counts2 = model(x2)
		bias_profile, _ = model.bias(x[:, :4])
		bias_profile2, _ = model.bias(x2[:, :4])

	assert seen[0] == (2, 4, 2114)
	assert profile.shape == (2, 1000) and counts.shape == (2, 1)
	assert not torch.allclose(counts, counts2)
	assert torch.equal(bias_profile, bias_profile2)


def test_train_step_uses_phylop_channel():
	from chrombpnet.model_wrappers import PhyloPChromBPNetWrapper

	wrapper = PhyloPChromBPNetWrapper(Namespace(**ARGS))
	x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 2114)), 4).float().transpose(1, 2)
	batch = {'onehot_seq': torch.cat([x, torch.randn(2, 1, 2114)], dim=1), 'profile': torch.rand(2, 1000)}
	wrapper.log_dict = lambda *a, **k: None
	opt = wrapper.configure_optimizers()
	assert opt.param_groups[0]['lr'] == 1e-3
	loss = wrapper._step(batch, 0, 'train')
	loss.backward()
	opt.step()

	assert torch.isfinite(loss)
	assert wrapper.model.model.iconv.weight.grad[:, 4].abs().sum() > 0


def test_factory_requires_phylop_bw():
	from chrombpnet.model_wrappers import create_model_wrapper

	with pytest.raises(ValueError, match="--phylop_bw"):
		create_model_wrapper(Namespace(model_type='phylop-chrombpnet', phylop_bw=None, **ARGS))
