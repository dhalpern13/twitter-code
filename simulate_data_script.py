import os
from collections import namedtuple
from itertools import product
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

from graph_generators import scale_down_by_following, preferential_attachment
from twitter import compute_limit_matrices, compute_engagement_vectors, maximize_engagement, compute_uniform_engagement, \
    opt_no_diversity

TYPES = [3, 5]
USERS = [15, 30, 100]
FOLLOWING_PER_USER = [2, 5, 10]
ITERATIONS = 200
TOTAL_DIVERSITY_STEPS = 10
GRAPH_SAMPLERS = {
    'scaled_gnp': scale_down_by_following,
    'preferential_attachment': preferential_attachment
}

directory = 'data'
divers_csv = 'diversity.csv'
uniform_csv = 'uniform.csv'


divers_columns = ['graph_id', 'users', 'types', 'graph_sampler', 'following_per_user',
                  'diversity_step', 'diversity_max_steps', 'engagement', 'prop_engagement']
uniform_columns = ['graph_id', 'users', 'types', 'graph_sampler', 'following_per_user',
                   'engagement', 'prop_engagement']

DiversityRow = namedtuple('diversity_row', field_names=divers_columns)
UniformRow = namedtuple('uniform_row', field_names=uniform_columns)

divers_path = os.path.join(directory, divers_csv)
uniform_path = os.path.join(directory, uniform_csv)

# Create files if they don't exist
os.makedirs(directory, exist_ok=True)
for file_path, cols in [(divers_path, divers_columns), (uniform_path, uniform_columns)]:
    if not os.path.exists(file_path):
        pd.DataFrame(columns=cols).to_csv(file_path, header=True, index=False)

for _ in tqdm(range(ITERATIONS)):
    for num_users, num_types, following_per_user, (sampler_name, graph_sampler) in product(USERS, TYPES,
                                                                                           FOLLOWING_PER_USER,
                                                                                           GRAPH_SAMPLERS.items()):
        # Give graph a unique id
        graph_id = uuid4()

        # Sample graph and compute related objects
        ps, type_matrices = graph_sampler(num_users, num_types, following_per_user)
        limit_matrices = compute_limit_matrices(type_matrices)
        engagement_vectors = compute_engagement_vectors(limit_matrices, ps)

        # Compute optimal engagement
        opt = opt_no_diversity(engagement_vectors)

        # Compute uniform engagement
        uniform_engagement = compute_uniform_engagement(engagement_vectors)
        prop_uniform_engagement = uniform_engagement / opt

        # Save every row to a file each iteration in case script is stopped
        uniform_row = UniformRow(graph_id=graph_id, users=num_users, types=num_types,
                                 graph_sampler=sampler_name,
                                 following_per_user=following_per_user, engagement=uniform_engagement,
                                 prop_engagement=prop_uniform_engagement)
        pd.DataFrame(data=[uniform_row]).to_csv(uniform_path, index=False, header=False, mode='a')

        for diversity_step in range(TOTAL_DIVERSITY_STEPS + 1):
            # E.g., if 10 steps and 3 types, compute on diversities 0/30,...,10/30
            diversity = diversity_step / TOTAL_DIVERSITY_STEPS / num_types
            engagement = maximize_engagement(limit_matrices, engagement_vectors, diversity=diversity)
            prop_engagement = engagement / opt
            divers_row = DiversityRow(graph_id=graph_id, users=num_users, types=num_types,
                                      graph_sampler=sampler_name,
                                      following_per_user=following_per_user, diversity_step=diversity_step,
                                      diversity_max_steps=TOTAL_DIVERSITY_STEPS, engagement=engagement,
                                      prop_engagement=prop_engagement)
            pd.DataFrame(data=[divers_row]).to_csv(divers_path, index=False, header=False, mode='a')
