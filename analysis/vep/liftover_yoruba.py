"""Lift the hg19 Yoruba dsQTL schema to hg38, keeping SNVs whose hg38 base matches an allele."""

import gzip
import sys

import pyfaidx

sys.path.insert(0, '/pollard/home/szhang20/arsenal-chrombpnet/analysis/vep/pylib')
from pyliftover import LiftOver

V = '/pollard/data/projects/shzhang/gpn/vep'
SRC = '/pollard/data/projects/shzhang/gpn/arsenal_download/yoruban_dsqtl_chrombpnet_schema.tsv'
OUT = f'{V}/data/yoruban_dsqtl_hg38_chrombpnet_schema.tsv'
COMPLEMENT = str.maketrans('ACGT', 'TGCA')

lo = LiftOver(f'{V}/data/hg19ToHg38.over.chain.gz')
fasta = pyfaidx.Fasta('/pollard/home/szhang20/.cache/genome/hg38/hg38.fa')
counts = dict(total=0, unmapped=0, other_chrom=0, minus_strand=0, allele_mismatch=0, kept=0)
with open(SRC) as src, open(OUT, 'w') as out:
	for line in src:
		chrom, pos, a1, a2, vid = line.rstrip('\n').split('\t')[:5]
		counts['total'] += 1
		# pyliftover takes and returns zero-based positions.
		hits = lo.convert_coordinate(chrom, int(pos) - 1)
		if not hits:
			counts['unmapped'] += 1
			continue
		new_chrom, new_pos0, strand, _ = hits[0]
		if new_chrom != chrom:
			counts['other_chrom'] += 1
			continue
		if strand == '-':
			counts['minus_strand'] += 1
			a1, a2 = a1.translate(COMPLEMENT), a2.translate(COMPLEMENT)
		if fasta[new_chrom][new_pos0].seq.upper() not in (a1, a2):
			counts['allele_mismatch'] += 1
			continue
		counts['kept'] += 1
		out.write(f'{new_chrom}\t{new_pos0 + 1}\t{a1}\t{a2}\t{vid}\n')
print(counts)
