"""ChromBPNet with per-base phyloP as a 5th input channel to the accessibility model."""

import torch.nn as nn

from .chrombpnet import ChromBPNet


class PhyloPChromBPNet(ChromBPNet):
	"""Accessibility model reads one-hot + phyloP (5 channels); the frozen bias model reads one-hot only."""

	def __init__(self, config, **kwargs):
		super().__init__(config, **kwargs)
		self.model.iconv = nn.Conv1d(5, config.n_filters, kernel_size=config.conv1_kernel_size, padding='valid')
		self.tf_style_reinit()

	def forward(self, x, **kwargs):
		"""x: (batch, 5, length), rows A/C/G/T then phyloP."""
		# BPNet transposes any input whose dim 1 is not 4, so pass (batch, length, 5).
		acc_profile, acc_counts = self.model(x.transpose(1, 2))
		bias_profile, bias_counts = self.bias(x[:, :4])

		y_profile = acc_profile + bias_profile
		y_counts = self._log(self._exp1(acc_counts) + self._exp2(bias_counts))

		# DO NOT SQUEEZE y_counts, as it is needed for running deep_lift_shap
		return y_profile.squeeze(1), y_counts
