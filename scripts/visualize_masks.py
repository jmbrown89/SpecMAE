"""Print mask statistics for quick inspection."""

from __future__ import annotations

import argparse

from specmae.spectral.mask import SpectralMaskConfig, make_keep_mask


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize mask statistics")
	parser.add_argument("--height", type=int, default=28)
	parser.add_argument("--width", type=int, default=28)
	parser.add_argument("--policy", choices=["high_freq", "low_freq", "random"], default="high_freq")
	parser.add_argument("--mask-ratio", type=float, default=0.6)
	args = parser.parse_args()

	mask = make_keep_mask(
		height=args.height,
		width=args.width,
		config=SpectralMaskConfig(policy=args.policy, mask_ratio=args.mask_ratio),
	)
	keep_ratio = mask.float().mean().item()
	print(f"shape={tuple(mask.shape)} keep_ratio={keep_ratio:.4f} mask_ratio={1.0 - keep_ratio:.4f}")


if __name__ == "__main__":
	main()

