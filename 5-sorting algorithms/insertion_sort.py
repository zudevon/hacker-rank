def insertion_sort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j-1]:
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1

if __name__ == '__main__':
    arr = [4, 5, 3, 2, 1]
    insertion_sort(arr)
    print(arr)