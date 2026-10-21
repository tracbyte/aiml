import heapq


def a_star(graph, start, goal, h):
    pq = [(h[start], 0, start, [start])]
    visited = {}

    while pq:
        f, g, node, path = heapq.heappop(pq)

        if node == goal:
            return path, g

        if node in visited and visited[node] <= g:
            continue
        visited[node] = g

        for neighbor, weight in graph.get(node, []):
            g_cost = g + weight
            f_cost = g_cost + h.get(neighbor, 0)
            heapq.heappush(pq, (f_cost, g_cost, neighbor, path + [neighbor]))
    return None, float("inf")


graph = {
    "A": [("B", 1), ("C", 3)],
    "B": [("D", 3), ("E", 1)],
    "C": [("F", 5)],
    "D": [("G", 2)],
    "E": [("G", 4)],
    "F": [("G", 1)],
    "G": [],
}
heuristics = {"A": 6, "B": 4, "C": 4, "D": 2, "E": 2, "F": 1, "G": 0}

path, cost = a_star(graph, "A", "G", heuristics)
print("Shortest Path:", path)
print("Total Cost:", cost)
