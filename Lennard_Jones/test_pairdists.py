import numpy as np
import random

def get_pairwise_distances(x, pairs):
    all_diffs = np.expand_dims(x, axis=1) - np.expand_dims(x, axis=0) # N * N * M
    diffs = np.array([all_diffs[i][j] for [i,j] in pairs])
    pairwise_distances = np.sqrt(np.sum(diffs**2, axis=-1)) # N * N

    return pairwise_distances


def generate_connected_pairs(M):
    if M < 3:
        return []

    # Create a fully connected graph
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]

    # Create an adjacency list
    adjacency_list = {i: [] for i in range(M)}
    for u, v in pairs:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    # Randomly remove edges while ensuring graph connectivity
    num_pairs_needed = 3 * M - 6
    while len(pairs) > num_pairs_needed:
        # Randomly select an edge to remove
        edge_to_remove = random.choice(pairs)

        # Remove the edge if it does not disconnect the graph
        if not disconnects_graph(adjacency_list, edge_to_remove):
            pairs.remove(edge_to_remove)
            u, v = edge_to_remove
            adjacency_list[u].remove(v)
            adjacency_list[v].remove(u)

    return pairs

def disconnects_graph(adjacency_list, edge):
    # Perform depth-first search to check connectivity after edge removal
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        node = stack.pop()
        for neighbor in adjacency_list[node]:
            if (node, neighbor) != edge and (neighbor, node) != edge and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    # If any node is not visited, the graph is disconnected
    return len(visited) != len(adjacency_list)



x = np.array([[0, 0, 0],
              [1, 0, 0],
              [2, 0, 0],
              [0, 1, 0],
              [1, 1, 0],
              [0, 2, 0]])

data = np.array([x, x])

pairs = np.array([[0,1], [4,5]])

# print(generate_connected_pairs(3))

print([get_pairwise_distances(data[i], pairs) for i in range(len(data))])
