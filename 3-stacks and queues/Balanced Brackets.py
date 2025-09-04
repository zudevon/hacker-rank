def isBalanced(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}

    for ch in s:
        if ch in '({[':
            stack.append(ch)
        else:  # closing bracket
            if not stack or stack[-1] != pairs[ch]:
                return "NO"
            stack.pop()

    return "YES" if not stack else "NO"

if __name__ == '__main__':
    s = input()
    print(isBalanced(s))