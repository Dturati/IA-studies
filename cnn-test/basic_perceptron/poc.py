a = (1, 2, 3)
b = (2, 2, 2)

res = zip(a, b)

for a, b in zip(a, b):
    print(a * b)
