def icecreamParlor(m, cost):
    seen = {}
    for i, c in enumerate(cost):
        if m - c in seen:
            return [seen[m - c] + 1, i + 1]  # +1 for 1-based index
        seen[c] = i

if __name__ == '__main__':
    money = int(input())
    cost = list(map(int, input().split()))
    r = icecreamParlor(money, cost)
    print(r)