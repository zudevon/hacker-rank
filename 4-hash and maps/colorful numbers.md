🧠 Complexity

Let n = number of digits.

Generating all subsequences = O(n²).

Each subsequence product computed in O(1) (running product).

Total Time = O(n²)

Space = O(n²) in worst case (for storing products).

🗣️ Interview Thinking-Aloud

If asked in an interview:

“We’re asked to check uniqueness of all subsequence products — that immediately suggests a set.”

“Brute force would be O(n³) if I recompute each product fresh, but I can optimize by maintaining a running product as I extend subsequences.”

“That brings it down to O(n²) time.”

“Space is also O(n²) since I might store up to n(n+1)/2 subsequences in the set.”