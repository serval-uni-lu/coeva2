print('hello')

def func(num):
    num = num+1
    a = num*5
    return a

for i in range(2):
    value = func(i)
    print(value)