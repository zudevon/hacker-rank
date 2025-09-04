🗣️ Interview Thinking-Aloud Tips

When solving this in an interview, you might say:

“I recognize this as a classic stack problem.”

“Every opening bracket gets pushed, and when I see a closing one, I check if it matches the last pushed bracket.”

“If at any point I mismatch or the stack is empty when I need to pop, it’s invalid.”

“At the end, the stack must also be empty — otherwise there were extra openings.”

“This runs in O(n) time with O(n) space in the worst case.”


⏱ Time Complexity

We process the string once, character by character:

For each character, we do either:

Push onto the stack (O(1)).

Pop from the stack and check (O(1)).

So every character requires only constant-time operations.

If the string length is n, total time = O(n).

👉 Time Complexity = O(n)

💾 Space Complexity

The stack stores only opening brackets.

In the worst case, the string could be all opening brackets (like "(((((((("), so the stack would hold n characters.

Each stack operation is O(1), but the maximum size grows with input length.

👉 Space Complexity = O(n) (worst case).