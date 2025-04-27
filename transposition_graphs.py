import sys
import argparse

# === Διαχείριση ορισμάτων ===
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transposition Graphs")
    parser.add_argument("s", type=int, help="number of 0 symbols")
    parser.add_argument("t", type=int, help="number of 1 symbols")
    parser.add_argument("method", choices=["graph", "dfs", "bts"], help="method to use")
    parser.add_argument("start", type=int, nargs="?", help="starting node (for dfs only)")
    return parser.parse_args()

# === Χωρίς itertools: δική μας υλοποίηση combinations ===
def combinations_custom(elements, k):
    if k == 0:
        return [[]]
    if not elements:
        return []
    with_first = [[elements[0]] + rest for rest in combinations_custom(elements[1:], k-1)]
    without_first = combinations_custom(elements[1:], k)
    return with_first + without_first

# === Δημιουργία κόμβων και γράφου ===
def generate_nodes(s, t):
    n = s + t
    positions = list(range(n))
    combos = combinations_custom(positions, t)
    return [sum(1 << (n-1-p) for p in comb) for comb in combos]

def generate_graph(nodes, s, t):
    graph = {node: [] for node in nodes}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if bin(nodes[i] ^ nodes[j]).count('1') == 2:
                graph[nodes[i]].append(nodes[j])
                graph[nodes[j]].append(nodes[i])
    return graph

# === DFS με ομογένεια και genlex ===
def to_indices(node, s, t):
    n_bits = s + t
    indices = []
    for i in range(n_bits):
        if (node >> (n_bits-1-i)) & 1:
            indices.append(n_bits-1-i)
    return indices

def is_homogeneous(u, v):
    return bin(u ^ v).count('1') == 2

def is_genlex_path(path_indices):
    for i in range(1, len(path_indices)):
        if path_indices[i-1] > path_indices[i]:
            return False
    return True

def dfs(graph, current, visited, path, s, t, solutions):
    visited.add(current)
    path.append(current)
    if len(path) == len(graph):
        solutions.append(list(path))
    else:
        for neighbor in sorted(graph[current], reverse=True):
            if neighbor not in visited:
                if is_homogeneous(current, neighbor):
                    test_path = path + [neighbor]
                    indices_path = [to_indices(p, s, t) for p in test_path]
                    if is_genlex_path(indices_path):
                        dfs(graph, neighbor, visited, path, s, t, solutions)
    visited.remove(current)
    path.pop()

# === Μέθοδος Balanced Ternary System (BTS) ===
def generate_sigma_j(s, t):
    sigmas = []
    for j in range(2**(s-1)):
        tau = []
        for k in range(s-1):
            if (j >> (s-2-k)) & 1:
                tau.append('-')
            else:
                tau.append('+')
        sigma = ['0']*t + ['-'] + tau
        sigmas.append(sigma)
    return sigmas

def next_balanced(s):
    s = s.copy()
    n = len(s)
    for i in range(n-1, -1, -1):
        if s[i] == '0':
            k = 0
            while i+k+1 < n and s[i+k+1] == '-':
                k += 1
            if i+k+1 < n and s[i+k+1] == '+':
                s[i] = '-'
                for j in range(1, k+1):
                    s[i+j] = '+'
                s[i+k+1] = '0'
                return s, True
            elif i > 0 and s[i-1] == '+':
                k = 0
                while i+k < n and s[i+k] == '-':
                    k += 1
                s[i-1] = '0'
                for j in range(k):
                    s[i+j] = '+'
                s[i+k] = '0'
                return s, True
    return s, False

def balanced_to_binary(s):
    return ['1' if c == '0' else '0' for c in s]

def binary_to_int(binary):
    return int(''.join(binary), 2)

def bts_method(s, t):
    sigmas = generate_sigma_j(s, t)
    all_paths = []
    for sigma in sigmas:
        path = []
        current = sigma.copy()
        seen = set()
        while True:
            state = tuple(current)
            if state in seen:
                break
            seen.add(state)
            bin_str = balanced_to_binary(current)
            node = binary_to_int(bin_str)
            path.append(node)
            current, ok = next_balanced(current)
            if not ok:
                break
        all_paths.append(path)
    return all_paths

# === Εμφάνιση ===
def int_to_binary_str(n, s, t):
    n_bits = s + t
    return format(n, f'0{n_bits}b')

def print_path(path, s, t):
    binaries = [int_to_binary_str(n, s, t) for n in path]
    print(binaries)
    indices_repr = [to_indices(n, s, t) for n in path]
    print(indices_repr)
    print(path)

# === Κύριο πρόγραμμα ===
if __name__ == "__main__":
    args = parse_arguments()
    nodes = generate_nodes(args.s, args.t)
    graph = generate_graph(nodes, args.s, args.t)

    if args.method == "graph":
        for node in sorted(graph.keys(), reverse=True):
            neighbors = sorted(graph[node], reverse=True)
            print(f"{node} -> {neighbors}")

    elif args.method == "dfs":
        solutions = []
        start_nodes = [args.start] if args.start is not None else nodes
        for start in start_nodes:
            if start not in nodes:
                continue
            dfs(graph, start, set(), [], args.s, args.t, solutions)
        for sol in solutions:
            print_path(sol, args.s, args.t)

    elif args.method == "bts":
        paths = bts_method(args.s, args.t)
        for path in paths:
            print_path(path, args.s, args.t)
