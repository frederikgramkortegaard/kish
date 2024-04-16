import math
if __name__ == "__main__":
    
    a = 290
    b = 1593
    while (b != 0):
        t = b
        b = a % b
        a = t
    print(a)