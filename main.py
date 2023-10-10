# def func1(func):
#     def wrapper(x, y):
#         divide = func(x, y)
#         print(divide)
#     return wrapper

# @func1
# def divided(x, y):
#     return x /y

# # divided(1, 2)

# x = func1(divided)

def plus_list(lst):
    for x in lst:
        yield x + 1
    print("Hello world")

for x in plus_list([1, 2, 3]):
    print(x)