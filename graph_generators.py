import networkx as nx
import numpy as np

from twitter import compute_type_matrices, compute_scaled_follower_matrix


def uniform_until_eig(users, types, following_per_user):
    """
    Keep sampling from Gnp until eigenvalues are at most 1.
    """
    edge_prob = following_per_user / users
    largest_eig = 1
    while largest_eig >= 1:
        follower_matrix = np.random.binomial(1, edge_prob, size=(users, users))
        np.fill_diagonal(follower_matrix, 0)
        ps = np.random.uniform(size=(types, 1, users))
        type_matrices = compute_type_matrices(follower_matrix, ps)
        largest_eig = max(max(np.absolute(np.linalg.eigvals(type_matrix))) for type_matrix in type_matrices)
        print(largest_eig)
    return ps, type_matrices


def scale_down_by_following(users, types, following_per_user):
    """
    Sample from Gnp and scale down transition probabilities by number of people following.
    """
    edge_prob = following_per_user / users
    follower_matrix = np.random.binomial(1, edge_prob, size=(users, users))
    np.fill_diagonal(follower_matrix, 0)

    ps = np.random.uniform(size=(types, 1, users))
    scaled_follower_matrix = compute_scaled_follower_matrix(follower_matrix)
    type_matrices = compute_type_matrices(scaled_follower_matrix, ps)

    return ps, type_matrices


def preferential_attachment(users, types, following_per_user):
    """
    Generate graph using directed preferential attachment.
    """
    undirected_graph = nx.barabasi_albert_graph(users, following_per_user,
                                                initial_graph=nx.complete_graph(following_per_user + 1))
    undirected_adjancecy_matrix = nx.to_numpy_array(undirected_graph)
    follower_matrix = np.triu(undirected_adjancecy_matrix)
    ps = np.random.uniform(size=(types, 1, users))
    type_matrices = compute_type_matrices(follower_matrix, ps)

    return ps, type_matrices


def preferential_attachment_scaled(users, types, following_per_user):
    """
    Generate graph using directed preferential attachment scaling the inputs by number of followers.
    """
    undirected_graph = nx.barabasi_albert_graph(users, following_per_user,
                                                initial_graph=nx.complete_graph(following_per_user + 1))
    undirected_adjancecy_matrix = nx.to_numpy_array(undirected_graph)
    follower_matrix = np.triu(undirected_adjancecy_matrix)
    ps = np.random.uniform(size=(types, 1, users))

    scaled_follower_matrix = compute_scaled_follower_matrix(follower_matrix)
    type_matrices = compute_type_matrices(scaled_follower_matrix, ps)

    return ps, type_matrices