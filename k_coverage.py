import sys
import threading
import heapq
from collections import deque

def solve():
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))

    reds = set(int(next(it)) - 1 for _ in range(m))

    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        adj[u].append((v, w))
        adj[v].append((u, w))

    # Dijkstra to compute distances from a start node
    def dijkstra(start):
        INF = 10**30
        dist = [INF] * n
        dist[start] = 0
        pq = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    # 1) 从任意红点出发，找到最远红点 A
    any_red = next(iter(reds))
    dist0 = dijkstra(any_red)
    A = max(reds, key=lambda x: dist0[x])

    # 2) 从 A 出发，找到最远红点 B
    distA = dijkstra(A)
    B = max(reds, key=lambda x: distA[x])

    # 3) BFS 获取 A->B 路径
    parent = [-1] * n
    q = deque([A])
    seen = [False] * n
    seen[A] = True
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)

    path = []
    cur = B
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    L = len(path)

    # 4) xi: 累积距离
    xi = [0] * L
    for i in range(1, L):
        # find weight from path[i-1] to path[i]
        u = path[i-1]; v = path[i]
        for to, w in adj[u]:
            if to == v:
                xi[i] = xi[i-1] + w
                break

    # 5) 多源Dijkstra 计算每个节点到最近路径节点的距离和索引
    INF = 10**30
    dist = [INF] * n
    idx = [-1] * n
    pq = []
    for i, u in enumerate(path):
        dist[u] = 0
        idx[u] = i
        heapq.heappush(pq, (0, u))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                idx[v] = idx[u]
                heapq.heappush(pq, (nd, v))

    # 6) 由各红点更新 wi
    wi = [0] * L
    for r in reds:
        pi = idx[r]
        wi[pi] = max(wi[pi], dist[r])

    # 7) 二分 + 贪心判断
    def can(D):
        intervals = []
        for i in range(L):
            if wi[i] > D:
                return False
            reach = D - wi[i]
            intervals.append((xi[i] - reach, xi[i] + reach))
        intervals.sort(key=lambda x: x[1])
        used = 0
        i = 0
        while i < L and used < 2:
            used += 1
            cover_pos = intervals[i][1]
            while i < L and intervals[i][0] <= cover_pos:
                i += 1
        return i >= L

    lo, hi = 0, xi[-1] + max(wi)
    ans = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if can(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    print(ans)

if __name__ == '__main__':
    solve()