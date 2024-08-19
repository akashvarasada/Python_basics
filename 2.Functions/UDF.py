
def iseven(a):
    if a%2==0:
        print(a,"is even.")
    else:
        print(a,"is odd.")


def maxofN(*b):
    l = [*b]
    return max(l)


def fibonaci(n):
    # fibonacci series
    # 0 1 1 2 3 5 8 13 21 34 55 ...
    a = 0
    b = 1

    for i in range(1,100):
        if a < 50:
            print(a, end=" ")
            a,b = b,a+b


def sumN(n):
    print((n*(n+1))/2)


def isprime(n):
    if n <= 1:
        return print(n, "is not a prime number")
    if n == 2:
        return print(n, "is a prime number")
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return print(n, "is not a prime number")
        else:
            return print(n, "is a prime number")

def factorial(n:int) -> str:
    fact = 1
    if n < 0:
        return print("Factorial does not exists for -ve numbers.")
    if n <= 1:
        return 1
    if n > 1:
        for i in range(2,n+1):
            fact = fact * i
        return print(f"{n}! is {fact}")
