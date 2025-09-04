# Why do I use two stacks

🌀 Why One Stack Isn’t Enough

A stack is LIFO (last in, first out), but a queue is FIFO (first in, first out).

If you only have one stack, when you try to dequeue, you always get the last inserted element, not the first.

So you can’t directly implement queue behavior with one stack.

⚖️ Why Two Stacks Work

Two stacks let us reverse the order twice:

stack_in: where we push all new enqueued elements.

Easy, O(1).

stack_out: used for dequeues and peeks.

When it’s empty, we pour everything from stack_in into stack_out.

This reverses the order → the oldest element ends up on top, ready to be dequeued.

👉 By separating the responsibilities:

stack_in = intake buffer.

stack_out = gives us FIFO order.

📝 Example

Operations: enqueue(1), enqueue(2), enqueue(3), dequeue()

Start: stack_in = [], stack_out = []

Enqueue 1 → stack_in = [1]

Enqueue 2 → stack_in = [1, 2]

Enqueue 3 → stack_in = [1, 2, 3]

Dequeue:

stack_out empty → move everything:

Pop 3 → push to stack_out → [3]

Pop 2 → push to stack_out → [3, 2]

Pop 1 → push to stack_out → [3, 2, 1]

Now dequeue → pop from stack_out → 1 ✅ (FIFO order)

🧠 Interview Way to Phrase It

If an interviewer asks “Why do we need two stacks?”, you’d say:

“A single stack can only give me LIFO order, but a queue requires FIFO. By using two stacks, I reverse the order of elements once when transferring from stack_in to stack_out. That way, the oldest element comes to the top of stack_out and can be dequeued first. This simulates the FIFO behavior of a queue using only LIFO operations.”