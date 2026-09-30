# Numerical protocol used for the manuscript

## Fixed AMDI parameters

The deterministic propagation parameters are stored in `configs/publication.json` and are not retuned between AMDI and Learned AMDI.

```text
alpha              0.010
beta               0.0002
tau                1e-7
sigma_c            0.12
refinement_decay   0.35
h                  0.60
initial_threshold  0.012
min_level           2
refine_fraction     0.10
coarsen_fraction    0.10
zeta                1e-6
```

## Learned policy

Actor input dimension: 6 local features.

Actor:

```text
6 -> 64 -> 64 -> 3
activation: tanh
```

Critic global descriptor: mean of 6 local features, componentwise maximum of 6 local features, global occupancy, and normalized time, giving 14 inputs.

Critic:

```text
14 -> 64 -> 64 -> 1
activation: tanh
```

## PPO

```text
gamma                   1.0
lambda_GAE              0.95
PPO clip                0.10
value coefficient       0.5
entropy coefficient     0.02
learning rate           5e-5
epochs/update            4
trajectories/update     16
updates                400
max gradient norm       0.5
```

Complete trajectories are collected first. GAE is evaluated per trajectory. Decision steps are pooled, advantages are normalized across the pooled rollout batch, and the pooled steps are randomly permuted in each optimization epoch.

## Reward

```text
lambda_err   1.0
lambda_occ   0.15  (selected policy)
lambda_sw    0.02
```

The Pareto study changes only `lambda_occ` over:

```text
0.03, 0.08, 0.15, 0.25, 0.30, 0.60
```

## Ordering

Every outer iteration follows:

```text
accepted state
  -> deterministic diffusion propagation
  -> feature construction
  -> adaptive decision
  -> constrained multiresolution update
  -> next accepted state
```

The final adaptive decision is not followed by an extra diffusion step.

## Reported quantities

`E_ref` and `C_rel` are terminal quantities. The switching measure reported in aggregate tables is the arithmetic mean over the six adaptive decisions. Terminal reconstructions are clipped pointwise to `[0,1]` before RMSE and SSIM are evaluated against the clean image.

## Section 5 controls

The retain-tree control keeps the threshold-initialized tree for all six outer iterations. The top-`K` and fixed-threshold controls refine once after the first propagation and retain thereafter. Their parameters are selected on validation images `100--103` with noise seed `20000 + image_seed` and frozen before testing.

The PPO and control frontiers use the same four test images and noise realizations. The `lambda_occ = 0.15` PPO point reuses the selected checkpoint. Transferred top-`K` and threshold parameters are not retuned at `64x64` or `128x128`.

## VAMPyR/MRCPP

The VAMPyR calculation is an order-five projection of the clean analytic target with precision `1e-3` and maximum depth eight. Its effective levels are qualitative localization indicators and are not numerically equated with Haar levels or treated as validation of the learned trajectory.
