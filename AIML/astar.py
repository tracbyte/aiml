import heapq

def a_star(graph, start, goal, h):
    pq = [(h[start], 0, start, [start])]
    
    while pq:
        _, g, node, path = heapq.heappop(pq)
        
        if node == goal:
            return path, g
            
        for neighbor, weight in graph[node]:
            heapq.heappush(pq, (g + weight + h[neighbor], g + weight, neighbor, path + [neighbor]))

graph = {
    'A': [('B', 1), ('C', 3)], 'B': [('D', 3), ('E', 1)], 
    'C': [('F', 5)], 'D': [('G', 2)], 'E': [('G', 4)], 
    'F': [('G', 1)], 'G': []
}
h = {'A': 6, 'B': 4, 'C': 4, 'D': 2, 'E': 2, 'F': 1, 'G': 0}

path, cost = a_star(graph, 'A', 'G', h)
print("Shortest Path:", path)
print("Total Cost:", cost)
