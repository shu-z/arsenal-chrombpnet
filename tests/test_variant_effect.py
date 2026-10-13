"""Variant scorer tests; run with ARSENAL_MODEL_DIR=~/arsenal pytest tests/test_variant_effect.py."""

import numpy as np
import pandas as pd
import pyBigWig
import pytest
import torch
from scipy.spatial.distance import jensenshannon

from chrombpnet.gpnstar import GPNStarEmbedder, _RC_TOKEN_TABLE, _TOKEN_TABLE, chunk_layout
from chrombpnet.variant_effect import FLANK, INPUT_LEN, VariantInputs, VariantScorer, jsd, softmax

N_SPECIES = 447
LENGTH = 6000
BASES = np.array(list('ACGT'))
COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def test_jsd_matches_scipy():
	rng = np.random.default_rng(0)
	p, q = softmax(rng.normal(size=(5, 1000))), softmax(rng.normal(size=(5, 1000)))
	assert np.allclose(jsd(p, q), [jensenshannon(a, b, base=2.0) for a, b in zip(p, q)])


@pytest.fixture
def genome(tmp_path):
	rng = np.random.default_rng(1)
	seq = ''.join(rng.choice(BASES, LENGTH))
	(tmp_path / 'g.fa').write_text('>chr1\n' + seq + '\n')
	return str(tmp_path / 'g.fa'), seq


def _variants(seq):
	"""Two SNVs whose allele1 is the reference base and allele2 another base."""
	pos = [2500, 3100]
	ref = [seq[p - 1] for p in pos]
	alt = [{'A': 'G', 'C': 'T', 'G': 'A', 'T': 'C'}[r] for r in ref]
	return pd.DataFrame({'chr': 'chr1', 'pos': pos, 'allele1': ref, 'allele2': alt, 'variant_id': ['v1', 'v2']})


def test_onehot_alleles_land_on_variant(genome):
	path, seq = genome
	v = _variants(seq)
	item = VariantInputs(v, 'chrombpnet', path)[0]
	for strand, idx in (('+', FLANK), ('-', INPUT_LEN - 1 - FLANK)):
		x1, x2 = item[strand]
		assert x1.shape == (2, 4, INPUT_LEN)
		for i, (a1, a2) in enumerate(zip(v['allele1'], v['allele2'])):
			if strand == '-':
				a1, a2 = COMP[a1], COMP[a2]
			assert BASES[x1[i, :, idx].argmax()] == a1 and BASES[x2[i, :, idx].argmax()] == a2
			assert (x1[i] != x2[i]).any(0).nonzero()[0].tolist() == [idx]
	start = v['pos'][0] - 1 - FLANK
	x1, _ = item['+']
	assert ''.join(BASES[x1[0].argmax(0)]) == seq[start:start + INPUT_LEN]
	rc = ''.join(COMP[b] for b in reversed(seq[start:start + INPUT_LEN]))
	assert ''.join(BASES[item['-'][0][0].argmax(0)]) == rc


def test_phylop_row_follows_strand(genome, tmp_path):
	path, seq = genome
	bw = pyBigWig.open(str(tmp_path / 'pos.bw'), 'w')
	bw.addHeader([('chr1', LENGTH)])
	bw.addEntries('chr1', 0, values=np.arange(LENGTH, dtype=np.float64).tolist(), span=1, step=1)
	bw.close()
	v = _variants(seq)
	item = VariantInputs(v, 'phylop-chrombpnet', path, phylop_bw=str(tmp_path / 'pos.bw'))[0]
	start = v['pos'][0] - 1 - FLANK
	positions = np.arange(start, start + INPUT_LEN)
	for x in item['+']:
		assert x.shape == (2, 5, INPUT_LEN) and np.array_equal(x[0, 4], positions)
	for x in item['-']:
		assert np.array_equal(x[0, 4], positions[::-1])
	# The DNA rows match the plain model's inputs exactly.
	plain = VariantInputs(v, 'chrombpnet', path)[0]
	for s in ('+', '-'):
		for a, b in zip(item[s], plain[s]):
			assert np.array_equal(a[:, :4], b)


class _FakeMSA:
	"""Deterministic random alignment bytes, tokenized like MSAReader.get."""

	def __init__(self):
		self.bytes = np.random.default_rng(2).choice(np.frombuffer(b'ACGT-', dtype=np.uint8), size=(LENGTH, N_SPECIES))

	def get(self, chrom, start, end, strand):
		b = self.bytes[start:end]
		return _RC_TOKEN_TABLE[b[::-1]] if strand == '-' else _TOKEN_TABLE[b]


class _Net(torch.nn.Module):
	def __init__(self, embedder):
		super().__init__()
		self.gpnstar_model = embedder


@pytest.fixture(scope='module')
def embedder():
	return GPNStarEmbedder('songlab/gpn-star-hg38-m447-25m', model_revision='386b4be7b22490d1fd1fab3c691bedbde046230d',
		chunk_size=256, num_layers_avg=6).eval()


def test_gpnstar_msa_alleles(genome):
	path, seq = genome
	v = _variants(seq)
	ds = VariantInputs(v, 'gpnstar-chrombpnet', path)
	ds._handles = {'msa': _FakeMSA()}
	item = ds[0]
	for strand, idx, table in (('+', FLANK, _TOKEN_TABLE), ('-', INPUT_LEN - 1 - FLANK, _RC_TOKEN_TABLE)):
		m1, m2 = item[strand]
		ref = np.stack([ds._handles['msa'].get('chr1', p - 1 - FLANK, p - 1 + FLANK, strand) for p in v['pos']])
		assert (m1 != m2).sum() == 2 and np.array_equal(np.delete(m1, idx, 1), np.delete(ref, idx, 1))
		assert m2[:, idx, 0].tolist() == table[[ord(a) for a in v['allele2']]].tolist()
		assert np.array_equal(m1[:, :, 1:], ref[:, :, 1:])


@pytest.mark.parametrize('strand', ['+', '-'])
def test_gpnstar_shortcut_matches_full_recompute(genome, embedder, strand):
	"""Re-embedding only the variant's chunk for allele2 equals embedding the whole allele2 window."""
	path, seq = genome
	ds = VariantInputs(_variants(seq), 'gpnstar-chrombpnet', path)
	ds._handles = {'msa': _FakeMSA()}
	m1, m2 = (torch.from_numpy(m) for m in ds[0][strand])
	scorer = object.__new__(VariantScorer)
	scorer.net, scorer.layout, scorer.width = _Net(embedder), chunk_layout(INPUT_LEN, 256), 256
	with torch.no_grad():
		e1, e2 = scorer.gpnstar_embeddings(m1, m2, strand)
		full = embedder(m2)
	assert torch.allclose(e2, full, atol=1e-5)
	changed = (e1 != e2).any(-1).any(0).nonzero().flatten()
	assert len(changed) and 929 <= changed.min() and changed.max() < 1185
