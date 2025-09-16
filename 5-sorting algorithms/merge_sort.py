def merge_sort(arr):
    # Base case: a list of size 0 or 1 is already sorted
    if len(arr) <= 1:
        return arr

    # Divide the list in half
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # Merge the two sorted halves
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0

    # Compare elements from left and right, add the smaller one
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Add any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result


if __name__ == '__main__':
    # Example usage:
    arr = [38, 27, 43, 3, 9, 82, 10]
    print("Original:", arr)
    print("Sorted:", merge_sort(arr))