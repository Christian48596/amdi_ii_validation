.PHONY: test verify validation figures figures-four-region reproduce

DEVICE ?= cpu
OUTDIR ?= IMG_TEST

test:
	pytest -q

verify:
	python verify_release.py

validation:
	python experiments/01_deterministic_backend_regression.py --config configs/publication.json
	python experiments/02_protocol_alignment_audit.py --config configs/publication.json
	python experiments/03_uniform_reference_check.py --config configs/publication.json

figures:
	python make_final_metric_figures.py --summary results/publication_summary_FINAL/publication_summary_FINAL.json --outdir $(OUTDIR)

figures-four-region:
	python make_four_region_figures.py --checkpoint checkpoints/selected/best_policy.pt --device $(DEVICE) --outdir $(OUTDIR)

reproduce:
	python reproduce_final_results.py --device $(DEVICE) --outdir reproduced_results
