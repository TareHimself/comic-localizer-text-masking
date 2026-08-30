"""Synthetic data generation and training for a comic text-mask segmenter.

Pipeline:

    label pages (cv-label)  ->  tm-prep-pages   -> cleaned .cvlabel
                            ->  tm-generate      -> mask dataset (crops)
                            ->  tm-pack          -> fast memmap pack
                            ->  tm-train         -> runs/<run>/checkpoints/best.pt
                            ->  tm-export-detect -> YOLO detection dataset

The core rendering primitive is `render.compose(surface, [(box, TextSpec)])`.
"""
