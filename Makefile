.PHONY: test verify validation section5 figures figures-four-region reproduce manifest

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

section5:
	python experiments/11_action_energy_audit.py --device $(DEVICE) --include-transfer
	python experiments/12_selector_controls.py --device $(DEVICE) --include-transfer

figures:
	python make_all_manuscript_figures.py --device $(DEVICE) --outdir $(OUTDIR)

figures-four-region:
	python make_four_region_figures.py --checkpoint checkpoints/selected/best_policy.pt --device $(DEVICE) --outdir $(OUTDIR)

reproduce:
	python reproduce_final_results.py --device $(DEVICE) --outdir reproduced_results

manifest:
	python tools/generate_manifest.py
