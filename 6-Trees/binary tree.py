# Node structure for a binary tree
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Simple Binary Tree
class BinaryTree:
    def __init__(self, root_value):
        self.root = Node(root_value)

    def insert_left(self, current_node, value):
        if current_node.left is None:
            current_node.left = Node(value)
        else:
            new_node = Node(value)
            new_node.left = current_node.left
            current_node.left = new_node

    def insert_right(self, current_node, value):
        if current_node.right is None:
            current_node.right = Node(value)
        else:
            new_node = Node(value)
            new_node.right = current_node.right
            current_node.right = new_node

    def inorder(self, node):
        if node:
            self.inorder(node.left)
            print(node.value, end=" ")
            self.inorder(node.right)

if __name__ == '__main__':
    # Example usage:
    tree = BinaryTree(1)
    tree.insert_left(tree.root, 2)
    tree.insert_right(tree.root, 3)
    tree.insert_left(tree.root, 4)
    tree.insert_right(tree.root.left, 5)

    print("Inorder traversal of tree:")
    tree.inorder(tree.root)