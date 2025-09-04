"""
pseudocode

function insertNodeAtPosition(head, data, position):
    newNode = Node(data)

    if position == 0:
        newNode.next = head
        return newNode

    current = head
    for i in range(position - 1):
        current = current.next

    newNode.next = current.next
    current.next = newNode

    return head

"""
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class llist:
    def __init__(self):
        self.head = None   # start with an empty list

    def append(self, data):
        """Add a new node to the end of the list"""
        new_node = Node(data)

        if self.head is None:   # if list is empty
            self.head = new_node
            return

        # otherwise, find the last node
        current = self.head
        while current.next:     # keep going until end
            current = current.next
        current.next = new_node

    def display(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")


def insertNodeAtPosition(head, data, position):
    new_node = Node(data)

    if position == 0:  # insert at head
        new_node.next = head
        return new_node

    current = head
    for _ in range(position - 1):
        current = current.next

    new_node.next = current.next
    current.next = new_node
    return head


if __name__ == '__main__':

    ll = llist()
    ll.append(16)
    ll.append(13)
    ll.append(17)
    ll.display()

    n = 3  # number of elements in the linked list
    data = 1  # node to be inserted
    position = 2  # position of data to be inserted into the list

    insertNodeAtPosition(ll.head, data, position)

    print(ll.display())