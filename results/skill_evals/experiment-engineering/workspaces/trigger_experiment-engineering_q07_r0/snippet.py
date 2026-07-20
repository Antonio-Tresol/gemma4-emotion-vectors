def pick_thirds(xs):
    a = []
    for i in range(len(xs)):
        b = xs[i * 3]
        a.append(b)
    return a


vals = [10, 20, 30, 40, 50]
print(pick_thirds(vals))
