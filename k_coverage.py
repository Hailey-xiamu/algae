import sys
import heapq
from collections import defaultdict, deque

def solve():
    n, m = map(int, input().split())
    reds = set(map(lambda x: int(x) - 1, input().split()))  # Convert to 0-indexed
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        u -= 1  # Convert to 0-indexed
        v -= 1
        adj[u].append((v, w))
        adj[v].append((u, w))
    
    # Dijkstra to find shortest distances from a source
    def dijkstra(start):
        dist = [float('inf')] * n
        dist[start] = 0
        pq = [(0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            
            for v, w in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        
        return dist
    
    # Find the diameter of red points (farthest pair)
    # Step 1: Find the farthest red point from an arbitrary red point
    any_red = next(iter(reds))
    dist_from_any = dijkstra(any_red)
    
    A = any_red
    max_dist = 0
    for red in reds:
        if dist_from_any[red] > max_dist:
            max_dist = dist_from_any[red]
            A = red
    
    # Step 2: Find the farthest red point from A
    dist_from_A = dijkstra(A)
    
    B = A
    max_dist = 0
    for red in reds:
        if dist_from_A[red] > max_dist:
            max_dist = dist_from_A[red]
            B = red
    
    # Reconstruct path from A to B
    def reconstruct_path():
        # BFS to find parent pointers
        parent = [-1] * n
        parent_weight = [0] * n
        queue = deque([A])
        visited = [False] * n
        visited[A] = True
        
        while queue:
            u = queue.popleft()
            for v, w in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    parent_weight[v] = w
                    queue.append(v)
        
        # Reconstruct path from B to A
        path = []
        current = B
        while current != -1:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return path, parent_weight
    
    path, parent_weight = reconstruct_path()
    L = len(path)
    
    # Compute cumulative distances along the path
    xi = [0] * L
    for i in range(1, L):
        xi[i] = xi[i-1] + parent_weight[path[i]]
    
    # For each node on path, compute max distance to red points in its subtree
    def compute_wi():
        wi = [0] * L
        path_set = set(path)
        
        for idx, node in enumerate(path):
            max_red_dist = 0
            
            # DFS from each neighbor not on the path
            for neighbor, edge_weight in adj[node]:
                if neighbor in path_set:
                    continue
                
                # BFS/DFS to find max distance to red points in this subtree
                stack = [(neighbor, edge_weight)]
                visited = set([node])  # Don't go back to path
                
                while stack:
                    u, dist_so_far = stack.pop()
                    if u in visited:
                        continue
                    visited.add(u)
                    
                    if u in reds:
                        max_red_dist = max(max_red_dist, dist_so_far)
                    
                    for v, w in adj[u]:
                        if v not in visited:
                            stack.append((v, dist_so_far + w))
            
            wi[idx] = max_red_dist
        
        return wi
    
    wi = compute_wi()
    
    # Binary search on the answer
    def can_cover_with_distance(D):
        # Create intervals for each path node
        intervals = []
        for i in range(L):
            if D < wi[i]:
                return False
            
            reach = D - wi[i]
            left = xi[i] - reach
            right = xi[i] + reach
            intervals.append((left, right))
        
        # Sort intervals by right endpoint
        intervals.sort(key=lambda x: x[1])
        
        # Greedy: place centers to cover all intervals
        centers_used = 0
        i = 0
        
        while i < len(intervals) and centers_used < 2:
            # Place a center at the rightmost position of current interval
            center_pos = intervals[i][1]
            centers_used += 1
            
            # Skip all intervals that this center covers
            while i < len(intervals) and intervals[i][0] <= center_pos:
                i += 1
        
        return i >= len(intervals)
    
    # Binary search
    left, right = 0, max(xi) + max(wi) if wi else max(xi)
    answer = right
    
    while left <= right:
        mid = (left + right) // 2
        if can_cover_with_distance(mid):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    print(answer)

if __name__ == '__main__':
    solve()