values_of_x = []
for tails in range(7):
    heads = 6 - tails
    if (tails > heads):
        diff = tails - heads
    else:
        diff = heads - tails
    values_of_x.append(diff)
    print(f"X({heads}H, {tails}T) = {diff}")

unique = []
for i in values_of_x:
    if i not in unique:
        unique.append(i)

print(f"unique possible values of x are: {unique[-1]}, {unique[-2]}, {unique[-3]}, {unique[-4]}")
