🗣️ Step 1 – Clarify the Problem

These are the kinds of questions to ask the interviewer (shows you’re detail-oriented):

“Is this a singly linked list or a doubly linked list?”
(In HackerRank, it’s usually singly.)

“Is the position always valid? For example, if the list has length 3, could I get position = 5?”
(They’ll often say “yes, it’s valid within bounds.”)

“Is the head ever null, meaning the list is empty?”
(If so, inserting at position 0 just makes the new node the head.)

“Is the position 0-indexed or 1-indexed?”
(On HackerRank, it’s 0-indexed.)

🗣️ Step 2 – Restate the Task

“So the goal is: given a linked list and a position, I need to create a new node with the given data, and link it in at that position.”

“If it’s at position 0, it becomes the new head. Otherwise, I walk the list until I get to position - 1, splice in the new node, and return the head.”

🗣️ Step 3 – Plan the Algorithm

“First, I’ll create a new node with the given data.”

“If position == 0: set newNode.next = head, return newNode.”

“Else, I’ll use a pointer current to traverse the list until I reach the node right before the insertion position.”

“Then I’ll set newNode.next = current.next and current.next = newNode.”

“Finally, return the head, since the head hasn’t changed.”

🗣️ Step 4 – Complexity Consideration

“Time complexity is O(n) in the worst case, because I may need to traverse the whole list if the position is near the end.”

“Space complexity is O(1), since I only allocate the new node.”

🗣️ Step 5 – Edge Cases to Call Out

“If the list is empty and position = 0, the new node becomes the head.”

“If position equals the list’s length, I’m basically inserting at the tail.”

“Since the problem guarantees valid positions, I don’t need to handle out-of-range errors.”

🗣️ Step 6 – Example Walkthrough

“Let’s test this with a simple example: Head = 16 → 13 → 7, data = 1, position = 2.
I traverse to node ‘13’ (position 1). I set newNode.next = 13.next, so newNode.next = 7. Then I set 13.next = newNode.
Final list: 16 → 13 → 1 → 7. That matches the expected output.”

🗣️ Step 7 – Write the Code

Then you’d go ahead and implement it (Python, Java, etc.) as we outlined earlier.

👉 This kind of structured “clarify → restate → plan → analyze → edge cases → dry run → code” flow shows interviewers that you’re thoughtful, methodical, and collaborative.