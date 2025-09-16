def balancedSplitExists(arr):
    if len(arr) < 2:
        return False
    arr.sort()
    total = sum(arr)
    prefix = 0
    for i in range(len(arr) - 1):
        prefix += arr[i]
        if prefix * 2 == total and arr[i] < arr[i + 1]:
            return True
    return False


if __name__ == '__main__':
    arr_1 = [2, 1, 2, 3, 7, 1, 2, 2]
    # [1, 1, 2, 2, 2, 2] + [3, 7]
    print(balancedSplitExists(arr_1))