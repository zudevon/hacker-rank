testGraph = {'0': ['3', '5', '9'],
             '1': ['6', '7', '4'],
             '2': ['10', '5', '6'],
             '3': ['0'],
             '4': ['1', '5', '8'],
             '5': ['2', '0', '4'],
             '6': ['1', '2'],
             '7': ['1'],
             '8': ['4'],
             '9': ['0'],
             '10': ['2'],
             }

def shortest_path(predecessorNode, start, end):
    path = [end]
    current = end
    while current != start:
        current = predecessorNode[current]
        path.append(current)

    path.reverse()
    return path

def bfsShortestPath(graph, startNode, endNode):
    visited = []
    queue = [startNode]
    predecessorNodes = {}

    while queue:
        currentNode = queue.pop(0)
        visited.append(currentNode)
        for neighbor in graph[currentNode]:
            if neighbor not in visited:
                queue.append(neighbor)
                predecessorNodes[neighbor] = currentNode
    print(shortest_path(predecessorNodes, startNode, endNode))

if __name__ == '__main__':
    bfsShortestPath(testGraph, '0', '6')