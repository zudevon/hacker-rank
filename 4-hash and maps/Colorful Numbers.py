def isColorful(number):
    digits = list(map(int, str(number)))
    print(digits)
    products = set()

    for i in range(len(digits)):
        product = 1
        for j in range(i, len(digits)):
            product *= digits[j]
            if product in products:
                return False
            products.add(product)
    return True

if __name__ == '__main__':
    print(isColorful(int(input())))