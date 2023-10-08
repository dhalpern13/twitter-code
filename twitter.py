import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import pyplot as plt
from tqdm import tqdm
gp.setParam(GRB.Param.LogToConsole, 0)
gp.setParam(GRB.Param.Method, 2)
gp.setParam(GRB.Param.Crossover, 0)
gp.setParam(GRB.Param.Threads, 1)


"""
Convention:
A[i, j] = 1 means i follows j, i.e., tweets flow from j to i.
Ps are row vectors, stored as TYPES x 1 x USERS
Engagement vectors are row vectors, stored as TYPES x 1 x USERS
Type and Limit matrices are stored as TYPESxUSERSxUSERS
Injection Bs are column vectors, stored as TYPES x USERS x 1
"""


def compute_scaled_follower_matrix(follower_matrix):
    """
    Compute a scaled down follower matrix where edges are scaled down by the number of other users a user follows.
    """
    num_following = follower_matrix.sum(axis=1, keepdims=True)
    to_scale = np.maximum(num_following, 1)  # Set 0s to 1s before dividing
    return follower_matrix / to_scale


def compute_type_matrices(follower_matrix, ps, scale_by_following=True):
    if scale_by_following:
        adjusted_follower_matrix = compute_scaled_follower_matrix(follower_matrix)
    else:
        adjusted_follower_matrix = follower_matrix
    # Numpy broadcasts to make this a TYPES x USERS x USERS
    return ps * adjusted_follower_matrix


def compute_limit_matrices(type_matrices):
    _, num_users, _ = type_matrices.shape
    return np.linalg.inv(np.identity(num_users) - type_matrices)


def compute_engagement_vectors(limit_matrices, ps):
    return ps @ limit_matrices


def compute_uniform_engagement(engagement_vectors):
    """Compute engagement from injecting uniform tweets to each user."""
    num_types, *_ = engagement_vectors.shape
    return engagement_vectors.sum() / num_types


def opt_no_diversity(engagement_vectors):
    """Optimum value with no diversity. Choose type with maximum engagement for each user."""
    return engagement_vectors.max(0).sum()


def lp_relaxation_lower_bound(type_matrices, engagement_vectors, diversity):
    """Compute Lower bound based on LP Relaxation. Only works if incoming edge sums are at most 1 for each user."""
    incoming_edge_sums = type_matrices.sum(2)  # TYPES x USERS
    spend_on_diversity = (1 - incoming_edge_sums) * diversity  # TYPES X USERS
    spend_on_diversity_per_user = spend_on_diversity.sum(0)  # USERS
    remaining_budget = 1 - spend_on_diversity_per_user  # USERS
    engagement_vectors_squeezed = engagement_vectors.squeeze(1)  # TYPES x USERS
    diversity_engagement = (spend_on_diversity * engagement_vectors_squeezed).sum()  # scalar
    optimal_engagement_per_user = engagement_vectors_squeezed.max(0)  # USERS
    remaining_engagement = (remaining_budget * optimal_engagement_per_user).sum()  # scalar
    return diversity_engagement + remaining_engagement


def lp_relaxation_lower_bound2(type_matrices, max_user_engagement, diversity, p_sum, num_types):
    """Compute Lower bound based on LP Relaxation. Only works if incoming edge sums are at most 1 for each user."""
    diversity_engagement = p_sum * diversity  # scalar
    incoming_edge_sums = type_matrices.sum((0, 2))  # USERS
    spend_on_diversity = (num_types - incoming_edge_sums) * diversity  # USERS
    remaining_budget = 1 - spend_on_diversity  # USERS
    remaining_engagement = (remaining_budget * max_user_engagement).sum()  # scalar
    return diversity_engagement + remaining_engagement


def maximize_engagement(limit_matrices, engagement_vectors, diversity):
    """Run LP to compute optimal engagement subject to diversity."""
    num_types, num_users, _ = limit_matrices.shape

    model = gp.Model()

    # num_types x num_users x 1 matrix representing injection to each user/type
    b_vars = model.addMVar((num_types, num_users, 1), lb=0, name='b')

    # Ensure all row sums are at most 1, i.e., total injection per user is at most 1.
    # (1, ..(TYPES).., 1) * B <= (1, ..(USERS).., 1)
    model.addConstr(np.ones(num_types) @ b_vars[:, :, 0] <= 1, 'feasible')

    for t, (limit_matrix, b_var) in enumerate(zip(limit_matrices, b_vars)):
        model.addConstr(limit_matrix @ b_var >= diversity, f'diversity-{t}')

    # sum_t engagement_t * b_t
    model.setObjective(
        gp.quicksum(engagement_vector @ b_var for engagement_vector, b_var in zip(engagement_vectors, b_vars)),
        GRB.MAXIMIZE)

    model.optimize()

    return model.getObjective().getValue()


def run_once(ps, type_matrices, diversity_steps=10, save_fig=None):
    """
    Run a single instance and plot the result.
    """
    print('Computing matrix inverses...')
    limit_matrices = compute_limit_matrices(type_matrices)

    print('Computing preliminaries...')
    engagement_vectors = compute_engagement_vectors(limit_matrices, ps)

    num_types, *_ = ps.shape

    diversities = (np.arange(diversity_steps) + 1) / diversity_steps / num_types
    theoretical_lower_bound = (1 - (num_types - 1) / num_types)

    print('Computing opt...')
    opt = opt_no_diversity(engagement_vectors)

    print('Computing uniform...')
    uniform_engagement = compute_uniform_engagement(engagement_vectors) / opt

    # print('Computing lower bound...')
    # lower_bound_engagement = lp_relaxation_lower_bound(type_matrices, engagement_vectors, 1 / num_types) / opt

    print('Computing diverse opts...')

    engagements = [maximize_engagement(limit_matrices, engagement_vectors, diversity) / opt for diversity in
                   tqdm(diversities)]
    print(engagements)
    print('Plotting...')
    plt.clf()
    plt.plot([0] + list(diversities), [1] + list(engagements))
    # plt.plot([0, 1 / num_types], [1, lower_bound_engagement])
    plt.plot([0, 1 / num_types], [1, uniform_engagement])
    plt.plot([0, 1 / num_types], [1, theoretical_lower_bound])
    # plt.legend(['optimal', 'lower bound', 'uniform', 'theoretical lower bound'])
    plt.legend(['optimal', 'uniform', 'theoretical lower bound'])
    if save_fig is None:
        plt.show()
    else:
        plt.savefig(save_fig)
    return engagements
