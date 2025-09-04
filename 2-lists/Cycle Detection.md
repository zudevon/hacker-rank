📖 Problem Summary

You’re given the head of a singly linked list, and you need to determine whether the list contains a cycle.

A cycle exists if, by following the next pointers, you eventually revisit a node you’ve already seen (instead of reaching None).

🗣️ Interview Thinking-Aloud Tips

If this came up in an interview, here’s how to talk it through:

“We need to detect a cycle in a singly linked list.”

“Naively, I could store all visited nodes in a set — O(n) time and O(n) space.”

“But there’s a famous optimal algorithm called Floyd’s Cycle Detection (tortoise and hare). If there’s a cycle, a fast pointer moving 2 steps and a slow pointer moving 1 step will eventually meet. If the fast pointer reaches the end (None), then there’s no cycle.”

“This gives me O(n) time and O(1) extra space, which is more efficient.”


