def recursive_sum(arr):
    """Recursive sum using slicing. Clear but slices the list each call (O(n^2) time overall)."""
    if not arr:          # base case: empty list -> sum = 0
        return 0
    return arr[0] + recursive_sum(arr[1:])

def recursive_sum_index(arr, i=0):
    """Recursive sum using an index to avoid slicing. O(n) time, O(n) recursion depth/stack. """
    if i >= len(arr):    # base case: past the end
        return 0
    return arr[i] + recursive_sum_index(arr, i+1)

if __name__ == '__main__':
    a = [3, 1, 4, 1, 5]
    print(recursive_sum(a))         # 14
    print(recursive_sum_index(a))   # 14

    print(recursive_sum([]))        # 0
    print(recursive_sum_index([]))  # 0
