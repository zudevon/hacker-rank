class QueueUsingTwoStacks:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def enqueue(self, x):
        self.stack_in.append(x)

    def dequeue(self):
        self._shift()
        self.stack_out.pop()

    def front(self):
        self._shift()
        return self.stack_out[-1]

    def _shift(self):
        # Move elements from stack_in to stack_out only when needed
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())


if __name__ == '__main__':
    q = QueueUsingTwoStacks()
    q.enqueue(42)
    q.dequeue()
    q.enqueue(14)
    q.front()
    q.enqueue(28)
    q.front()
    q.enqueue(68)
    q.enqueue(70)
    q.dequeue()
    q.dequeue()
    print(q.front())

"""
Ask why there is even a second stack

easy solution

query =[]
number_of_queries = int(input())
for q in range(number_of_queries):
    operation = input().split()
    if len(operation) == 2:
        query.append(operation[1])
    elif len(operation) == 1 and operation[0] != '3':
        query.remove(query[0])
    elif operation[0] == '3':
        print(query[0])

"""