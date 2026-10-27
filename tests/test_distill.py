"""Tests for the distilled GPN-Star + ChromBPNet integration."""

import os
from argparse import Namespace

import numpy as np
import pytest
import torch

from chrombpnet.dataset import DistillDataset
from chrombpnet.distill import DistillEmbedder, fixed_rms_norm, rc_tokens, tokenize

MODEL = '/pollard/data/projects/shzhang/gpn-star-distill/models/gpn-star-distill-149m-2head-4kb'
needs_model = pytest.mark.skipif(not os.path.exists(f'{MODEL}/model.safetensors'), reason='distilled weights not downloaded')


def test_tokenize_and_rc():
	assert tokenize('ACGTNacgtR').tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
	assert rc_tokens(tokenize('AACGN')).tolist() == tokenize('NCGTT').tolist()


@pytest.fixture
def fasta(tmp_path):
	rng = np.random.default_rng(0)
	seq = ''.join(rng.choice(list('ACGT'), 10_000))
	path = tmp_path / 'genome.fa'
	path.write_text('>chrT\n' + '\n'.join(seq[i:i + 60] for i in range(0, len(seq), 60)) + '\n')
	return str(path), seq


def _dataset(path, context):
	ds = object.__new__(DistillDataset)
	ds.genome_fasta, ds.context, ds._fasta = path, context, None
	return ds


@pytest.mark.parametrize('strand', ['+', '-'])
def test_tokens_center_crop_matches_window(fasta, strand):
	path, seq = fasta
	ds = _dataset(path, 4096)
	start = 3000
	tok = ds.tokens('chrT', start, start + 2114, strand)
	assert tok.shape == (4096,)
	window = tokenize(seq[start:start + 2114])
	expected = rc_tokens(window) if strand == '-' else window
	# The central 2,114 of the 4,096 tokens are exactly the ChromBPNet window, on its strand.
	off = (4096 - 2114) // 2
	assert np.array_equal(tok[off:off + 2114], expected)


def test_tokens_pad_past_chromosome_ends(fasta):
	path, seq = fasta
	tok = _dataset(path, 4096).tokens('chrT', 0, 2114, '+')
	assert (tok[:991] == 4).all() and np.array_equal(tok[991:991 + 2114], tokenize(seq[:2114]))


@pytest.fixture(scope='module')
def embedder():
	return DistillEmbedder(MODEL, crop=200).eval()


def _tokens(length=600, seed=1):
	return torch.randint(0, 4, (2, length), generator=torch.Generator().manual_seed(seed))


@needs_model
def test_shared_final_layer(embedder):
	tok = _tokens()
	with torch.no_grad():
		got = embedder(tok)
		ref = fixed_rms_norm(embedder.model.embed(tok, head=None))[:, 200:400]
	assert got.shape == (2, 200, 1024)
	torch.testing.assert_close(got, ref)


@needs_model
def test_m447_matches_head_norm(embedder):
	tok = _tokens()
	embedder.representation = 'm447'
	try:
		with torch.no_grad():
			got = embedder(tok)
			# The m447 head's own RMSNorm module applied to the raw final layer.
			ref = embedder.model.embed(tok, head='m447')[:, 200:400]
	finally:
		embedder.representation = 'shared'
	torch.testing.assert_close(got, ref.float())


@needs_model
def test_last_layers_average(embedder):
	tok = _tokens()
	embedder.num_layers_avg = 6
	try:
		with torch.no_grad():
			got = embedder(tok)
			h, hs = embedder.model.embed_norm(embedder.model.embedding(tok)), []
			for layer in embedder.model.layers:
				h = layer(h)
				hs.append(h)
			ref = torch.stack([fixed_rms_norm(x)[:, 200:400] for x in hs[-6:]]).mean(0)
	finally:
		embedder.num_layers_avg = 1
	torch.testing.assert_close(got, ref)


def test_head_representation_needs_final_layer():
	if not os.path.exists(f'{MODEL}/model.safetensors'):
		pytest.skip('distilled weights not downloaded')
	with pytest.raises(ValueError):
		DistillEmbedder(MODEL, representation='m447', num_layers_avg=6)


@needs_model
def test_train_step_updates_only_chrombpnet():
	from chrombpnet.model_wrappers import DistillChromBPNetWrapper
	args = Namespace(
		alpha=1.0, beta=1.0, verbose=False,
		out_dim=1000, n_filters=32, n_layers=2, conv1_kernel_size=21, profile_kernel_size=75,
		n_outputs=1, n_control_tracks=0, profile_output_bias=True, count_output_bias=True,
		input_len=2114, input_embedding_dim=1024, num_layers_avg=1,
		distill_model=MODEL, distill_representation='shared', distill_context=2200, distill_attn='sdpa',
	)
	wrapper = DistillChromBPNetWrapper(args)
	before = {k: v.clone() for k, v in wrapper.model.distill_model.state_dict().items()}
	batch = {
		'onehot_seq': torch.nn.functional.one_hot(torch.randint(0, 4, (2, 2114)), 4).float().transpose(1, 2),
		'tokens': torch.randint(0, 4, (2, 2200)),
		'profile': torch.rand(2, 1000),
	}
	opt = torch.optim.Adam([p for p in wrapper.parameters() if p.requires_grad], lr=1e-3)
	loss = wrapper._step(batch, 0, mode='train')
	loss.backward()
	assert wrapper.model.model.iconv.weight.grad is not None
	assert all(p.grad is None for p in wrapper.model.distill_model.parameters())
	opt.step()
	for k, v in wrapper.model.distill_model.state_dict().items():
		assert torch.equal(v, before[k]), k
