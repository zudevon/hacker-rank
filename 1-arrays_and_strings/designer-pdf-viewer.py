import os


#
# Complete the 'designerPdfViewer' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY h
#  2. STRING word
#

# questions to ask


def designerPdfViewer(h, word):
    # Write your code here
    chars = 'abcdefghijklmnopqrstuvwxyz'
    h_chars = {}
    for index, char in enumerate(chars):
        h_chars[chars[index]] = h[index]

    h_list = []
    for c in word:
        h_list.append(h_chars[c])

    return max(h_list) * len(word)


if __name__ == '__main__':

    h = list(map(int, input().rstrip().split()))

    word = input().strip()

    result = designerPdfViewer(h, word)

    print(result)
