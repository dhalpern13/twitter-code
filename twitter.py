import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import pyplot as plt

gp.setParam(GRB.Param.LogToConsole, 0)

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
    scaling = np.maximum(num_following, 1)  # Turn 0s into 1s before dividing
    return follower_matrix / scaling

def compute_type_matrices(follower_matrix, ps, scale_by_following=True):
    if scale_by_following:
        follower_matrix = compute_scaled_follower_matrix(follower_matrix)
    # Numpy broadcasts to make this a TYPES x USERS x USERS
    return ps * follower_matrix


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
    remaining_budget = 1 - spend_on_diversity.sum(0)  # USERS
    diversity_engagement = (spend_on_diversity[:, np.newaxis, :] * engagement_vectors).sum()
    optimal_engagement_per_user = engagement_vectors.max(0).squeeze()  # USERS
    remaining_engagement = (remaining_budget * optimal_engagement_per_user).sum()
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


def run_once(ps, type_matrices, diversity_steps=10):
    """
    Run a single instance and plot the result.
    """
    limit_matrices = compute_limit_matrices(type_matrices)
    engagement_vectors = compute_engagement_vectors(limit_matrices, ps)

    num_types, *_ = ps.shape

    diversities = np.arange(diversity_steps + 1) / diversity_steps / num_types
    theoretical_lower_bound = (1 - diversities * (num_types - 1))

    opt = opt_no_diversity(engagement_vectors)

    uniform_engagement = compute_uniform_engagement(engagement_vectors) / opt

    engagements = [maximize_engagement(limit_matrices, engagement_vectors, diversity) / opt for diversity in
                   diversities]

    lower_bound_engagement = lp_relaxation_lower_bound(type_matrices, engagement_vectors, 1 / num_types) / opt

    plt.plot(diversities, engagements)
    plt.plot([0, 1 / num_types], [1, lower_bound_engagement])
    plt.plot([0, 1 / num_types], [1, uniform_engagement])
    plt.plot(diversities, theoretical_lower_bound)
    print((1 - np.array(engagements[1:])) / diversities[1:] / (num_types - 1))
    print(1 - lower_bound_engagement * num_types / (num_types - 1))
    plt.legend(['optimal', 'lower bound', 'uniform', 'theoretical lower bound'])
    plt.show()


def main():
    users = 10

    follower_matrix = np.random.binomial(1, .5, (users, users))
    np.fill_diagonal(follower_matrix, 0)

    ps = np.ones((3, 1, users)) * np.array([.9, .4, .3])[:, np.newaxis, np.newaxis]
    type_matrices = compute_type_matrices(follower_matrix, ps)

    run_once(ps, type_matrices, diversity_steps=10)


if __name__ == '__main__':
    main()
