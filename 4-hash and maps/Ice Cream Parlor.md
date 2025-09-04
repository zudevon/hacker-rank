⚡ Solution Approaches
1. Brute Force (O(n²))

Check all pairs (i, j) until you find one that sums to m.
Works but slow for large input.

def icecreamParlor(m, cost):
    n = len(cost)
    for i in range(n):
        for j in range(i+1, n):
            if cost[i] + cost[j] == m:
                return [i+1, j+1]

2. Hash Map (Efficient, O(n))

Store costs in a dictionary as you go.
For each cost, check if m - cost[i] exists.

def icecreamParlor(m, cost):
    seen = {}
    for i, c in enumerate(cost):
        if m - c in seen:
            return [seen[m - c] + 1, i + 1]  # +1 for 1-based index
        seen[c] = i


Time: O(n)

Space: O(n)

3. Two-Pointer (After Sorting)

If order of indices doesn’t matter, sort costs and use two pointers.
But since HackerRank expects original indices, hashmap approach is preferred.

🧠 Interview Thinking-Aloud

If this comes up in an interview, you’d say:

“I need two numbers that sum to m. That’s the Two-Sum problem.”

“Brute force is O(n²), but I can do better using a hash map to store complements.”

“At each step, I check if the complement exists. If yes, I found the answer. If not, I store the current value and its index.”

“That makes the solution O(n) time and O(n) space.”