product_list = ['A01-10-25', 'B01-40-10', 'C01-50-30', 'D01-10-15']

def avg_capacity(list_1):
    sum = 0.00
    for i in list_1:
        capacity = float(i.split("-")[1])
        sum+=capacity
    return sum/len(list_1)
print(avg_capacity(product_list))



