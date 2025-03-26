from acoustools.Utilities import batch_list


x = range(100)
y = range(100,200)

for b in batch_list(x):
    print(b)

for a,b in zip(batch_list(x), batch_list(y)):
    print(a,b)


for a,b in zip(batch_list(x), batch_list(None)):
    print(a,b)