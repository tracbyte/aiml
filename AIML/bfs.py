from collections import deque

# Define the graph using a dictionary where keys are nodes
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

def bfs(graph, start, goal):
    # Queue stores paths, initialized with the start node
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        # Return the path if we reached the goal
        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            # Generate and queue new paths for each neighbor
            for neighbour in graph[node]:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

    return None

print("Path:", bfs(graph, 'A', 'F'))