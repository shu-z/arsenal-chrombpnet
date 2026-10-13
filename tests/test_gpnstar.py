"""GPN-Star + ChromBPNet tests; run with ARSENAL_MODEL_DIR=~/arsenal pytest tests/test_gpnstar.py."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from chrombpnet.gpnstar import GPNStarChromBPNet, GPNStarEmbedder, chunk_layout
from chrombpnet.model_config import GPNStarChromBPNetConfig

N_SPECIES = 447
# name: (repo, revision, hidden_size, num_layers)
MODELS = {
	"25m": ("songlab/gpn-star-hg38-m447-25m", "386b4be7b22490d1fd1fab3c691bedbde046230d", 512, 8),
	"85m": ("songlab/gpn-star-hg38-m447-85m", "ac9ac24320fcc483f1f498338c318d8533ccf7d4", 768, 12),
}


@pytest.mark.parametrize("seq_len,chunk", [(2114, 256), (2114, 350), (1000, 256), (256, 256), (100, 256), (2114, 2114)])
def test_chunk_layout_tiles_once(seq_len, chunk):
	covered = np.zeros(seq_len, dtype=int)
	for w, a, b in chunk_layout(seq_len, chunk):
		assert 0 <= w and w + min(chunk, seq_len) <= seq_len
		assert w <= a < b <= w + min(chunk, seq_len)
		covered[a:b] += 1
	assert (covered == 1).all()


def _position_stub(embedder):
	"""Replace the model with one whose 'embedding' is each position's token."""
	embedder.embed_chunks = lambda msa: msa[:, :, :1].float()
	return embedder


def test_stitching_puts_every_position_back():
	embedder = _position_stub(object.__new__(GPNStarEmbedder))
	torch.nn.Module.__init__(embedder)
	embedder.chunk_size, embedder.chunk_batch_size = 256, 7  # odd micro-batch on purpose
	msa = torch.zeros(3, 2114, 2, dtype=torch.long)
	msa[:, :, 0] = torch.arange(2114)
	out = embedder(msa)
	assert out.shape == (3, 2114, 1)
	assert torch.equal(out[..., 0].long(), msa[:, :, 0])


def test_matches_arsenal_chunking():
	"""Same chunk boundaries as ArsenalChromBPNet.get_embeddings, via a position-in-chunk stub."""
	from chrombpnet.chrombpnet import ArsenalChromBPNet

	seq_len, chunk = 2114, 256
	tokens = torch.arange(seq_len).repeat(2, 1)

	arsenal = object.__new__(ArsenalChromBPNet)
	torch.nn.Module.__init__(arsenal)
	arsenal.seq_input_size, arsenal.arsenal_input_size, arsenal.num_layers_avg = seq_len, chunk, 4
	arsenal.get_avg_embeddings = lambda t, n: torch.stack([t, torch.arange(t.shape[1]).expand_as(t)], -1).float()
	expected = arsenal.get_embeddings(tokens)

	embedder = object.__new__(GPNStarEmbedder)
	torch.nn.Module.__init__(embedder)
	embedder.chunk_size, embedder.chunk_batch_size = chunk, 1000
	embedder.embed_chunks = lambda m: torch.stack([m[:, :, 0], torch.arange(m.shape[1]).expand(m.shape[:2])], -1).float()
	got = embedder(tokens[:, :, None])
	assert torch.equal(got, expected)


@pytest.fixture(scope="module", params=list(MODELS))
def embedder(request):
	repo, rev, _, _ = MODELS[request.param]
	return GPNStarEmbedder(repo, model_revision=rev, chunk_size=256, num_layers_avg=4).eval()


def _random_msa(batch, length, seed=0):
	g = torch.Generator().manual_seed(seed)
	return torch.randint(1, 5, (batch, length, N_SPECIES), generator=g, dtype=torch.uint8)


def test_real_model_shapes(embedder):
	msa = _random_msa(2, 600)
	with torch.no_grad():
		out = embedder(msa)
	assert out.shape == (2, 600, embedder.hidden_size)
	assert torch.isfinite(out).all()


def test_last_layers_average(embedder):
	msa = _random_msa(1, 256).long()
	with torch.no_grad():
		hs = embedder.model(
			input_ids=msa[:, :, :1], source_ids=msa,
			target_species=torch.zeros(1, 1, dtype=torch.long), output_hidden_states=True,
		).hidden_states
		got = embedder.embed_chunks(msa)
	assert len(hs) == embedder.model.config.num_hidden_layers + 1
	torch.testing.assert_close(got, torch.stack(hs[-4:]).mean(0)[:, :, 0])


@pytest.mark.parametrize("name", list(MODELS))
def test_train_step_updates_only_chrombpnet(name):
	from chrombpnet.model_wrappers import GPNStarChromBPNetWrapper

	repo, rev, hidden, n_layers = MODELS[name]
	args = Namespace(
		alpha=1.0, beta=1.0, verbose=False,
		out_dim=1000, n_filters=64, n_layers=2, conv1_kernel_size=21, profile_kernel_size=75,
		n_outputs=1, n_control_tracks=0, profile_output_bias=True, count_output_bias=True,
		input_len=2114, input_embedding_dim=hidden, num_layers_avg=n_layers // 2,
		gpnstar_model=repo, gpnstar_revision=rev, gpnstar_input_size=256,
		gpnstar_chunk_batch_size=64, finetune_gpnstar=False,
	)
	wrapper = GPNStarChromBPNetWrapper(args)
	gpn_before = {k: v.clone() for k, v in wrapper.model.gpnstar_model.state_dict().items()}

	batch = {
		'onehot_seq': torch.nn.functional.one_hot(torch.randint(0, 4, (2, 2114)), 4).float().transpose(1, 2),
		'msa': _random_msa(2, 2114),
		'profile': torch.rand(2, 1000),
	}
	wrapper.log_dict = lambda *a, **k: None
	opt = wrapper.configure_optimizers()
	wrapper.train()
	loss = wrapper._step(batch, 0, 'train')
	loss.backward()
	opt.step()

	assert torch.isfinite(loss)
	assert all(p.grad is None for p in wrapper.model.gpnstar_model.parameters())
	assert wrapper.model.model.iconv.weight.grad is not None
	assert not wrapper.model.gpnstar_model.training
	for k, v in wrapper.model.gpnstar_model.state_dict().items():
		assert torch.equal(v, gpn_before[k]), k


@pytest.mark.parametrize("name", list(MODELS))
def test_config_rejects_wrong_embedding_dim(name):
	repo, rev, hidden, _ = MODELS[name]
	config = GPNStarChromBPNetConfig(input_embedding_dim=hidden + 1, gpnstar_model=repo, gpnstar_revision=rev)
	with pytest.raises(ValueError, match=f"hidden_size={hidden}"):
		GPNStarChromBPNet(config)


def test_window_from_coords():
	from chrombpnet.dataset import GPNStarDataset

	ds = object.__new__(GPNStarDataset)
	ds.inputlen = 2114
	# random_crop leaves peaks at the window start; non-peaks keep the summit.
	ds.cur_coords = np.array([
		['chr1', '1000', 'f', '1'],
		['chr2', '5000', 'r', '0'],
	])
	assert ds.window(0) == ('chr1', 1000, 3114, '+')
	assert ds.window(1) == ('chr2', 5000 - 1057, 5000 + 1057, '-')


def test_window_matches_cropped_sequence():
	"""window() matches onehot_seq after the repo's own jitter and revcomp."""
	from chrombpnet.dataset import crop_revcomp_data
	from chrombpnet.dataset import GPNStarDataset

	rng = np.random.default_rng(0)
	inputlen, outputlen, jitter = 2114, 1000, 500
	genome = rng.integers(0, 4, 20_000)
	onehot = np.eye(4)[genome]

	def load(summits, width, out_width, is_peak):
		seqs = np.stack([onehot[s - width // 2:s + width // 2] for s in summits])
		cts = np.zeros((len(summits), out_width))
		coords = np.array([['chr1', s, 'f', is_peak] for s in summits])
		return seqs, cts, coords

	peak_summits = rng.integers(3000, 17000, 40)
	nonpeak_summits = rng.integers(3000, 17000, 400)
	peaks = load(peak_summits, inputlen + 2 * jitter, outputlen + 2 * jitter, 1)
	nonpeaks = load(nonpeak_summits, inputlen, outputlen, 0)

	seqs, _, coords = crop_revcomp_data(*peaks, *nonpeaks, inputlen, outputlen,
		add_revcomp=True, negative_sampling_ratio=0.5)

	ds = object.__new__(GPNStarDataset)
	ds.inputlen, ds.cur_coords = inputlen, coords
	assert {c[2] for c in coords} == {'f', 'r'} and {c[3] for c in coords} == {'0', '1'}
	for i in range(len(seqs)):
		chrom, start, end, strand = ds.window(i)
		expected = onehot[start:end]
		if strand == '-':
			expected = expected[::-1, ::-1]
		assert np.array_equal(seqs[i], expected), (i, coords[i])


def test_msa_reader_matches_genome_msa(tmp_path):
	"""MSAReader equals GenomeMSA.get_msa across stores, strands, odd bytes and chrom edges."""
	import zarr
	from gpn.star.data import GenomeMSA
	from chrombpnet.gpnstar import MSAReader

	rng = np.random.default_rng(0)
	alphabet = np.frombuffer(b"ACGTacgtNn-", dtype="S1")
	stores = {}
	for n in (5, 3):
		g = zarr.open_group(tmp_path / str(n) / "all.zarr", mode="w", zarr_format=2)
		arr = rng.choice(alphabet, size=(3000, n))
		g.create_array("7", shape=arr.shape, chunks=(512, n), dtype="S1")[:] = arr
		stores[n] = GenomeMSA(str(tmp_path / str(n) / "all.zarr"), n_species=n)

	reader = MSAReader(str(tmp_path))
	assert reader.n_species == 8
	for start, end in [(0, 2114), (800, 2914), (-50, 600), (2500, 3200)]:
		for strand in "+-":
			expected = np.concatenate(
				[stores[n].get_msa("7", start, end, strand=strand, tokenize=True) for n in (5, 3)],
				axis=-1,
			)
			got = reader.get("chr7", start, end, strand)
			assert got.dtype == np.uint8
			np.testing.assert_array_equal(got, expected)
