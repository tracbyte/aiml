from collections import deque

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None

graph = {
    'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['F'], 
    'D': [], 'E': ['G'], 'F': ['G'], 'G': []
}

print("BFS Path:", bfs(graph, 'A', 'G'))
