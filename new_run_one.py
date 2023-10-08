from itertools import product
from sys import argv

import numpy as np
from tqdm import tqdm

from twitter import compute_type_matrices, compute_limit_matrices, compute_engagement_vectors, \
    opt_no_diversity, maximize_engagement

datasets = ['guncontrol', 'obamacare']
all_ps = ['mode', 'beta-1-100-1', 'beta-1-100-2']
set_scales = list(product([.1, .3, .9], [True])) + list(product([1, 3, 10, 30], [False]))


param_combos = [(dataset, ps, set_scale) for dataset, ps, set_scale in product(datasets, all_ps, set_scales)]
instance = int(argv[1])
dataset, ps_name, (modify_val, is_set) = param_combos[instance]
is_set_str = "set" if is_set else "scale"
ps = np.load(f'data/numpy_arrays/{dataset}-{ps_name}.npy')
graph = np.load(f'data/numpy_arrays/{dataset}-graph.npy')
print(f'{instance}: {dataset}, {ps_name}, {is_set_str}, {modify_val}')

if is_set:
    unscaled_type_matrices = compute_type_matrices(graph, ps, scale_by_following=False)
    incoming_weight = unscaled_type_matrices.sum(axis=(0, 2))
    to_scale = np.divide(modify_val, incoming_weight, where=incoming_weight != 0, out=np.zeros_like(incoming_weight))
    type_matrices = unscaled_type_matrices * to_scale[None, :, None]
else:
    scaled_ps = np.minimum(ps * modify_val, 0.99)
    type_matrices = compute_type_matrices(graph, scaled_ps, scale_by_following=True)
print('Computing matrix inverses...')
limit_matrices = compute_limit_matrices(type_matrices)

print('Computing preliminaries...')
engagement_vectors = compute_engagement_vectors(limit_matrices, ps)

print('Computing opt...')
# opt = opt_no_diversity(engagement_vectors)
opt = 30

num_types, *_ = ps.shape

for step in tqdm(range(1, 11)):
    diversity = step / 40

    print(f'Computing divers opts {step}...')

    # engagement = maximize_engagement(limit_matrices, engagement_vectors, diversity)
    engagement = 20
    print(engagement)

    with open(f'diversities/{dataset}-{ps_name}-{is_set_str}-{modify_val}-{step}.txt', 'w') as f:
        f.write(f'{engagement}\n{opt}\n{engagement / opt}')