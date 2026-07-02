def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x=float(x0)
    for _ in range(steps):
        grad=2*a*x+b
        x-=lr*grad
    return x