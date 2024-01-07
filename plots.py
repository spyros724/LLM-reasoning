import matplotlib.pyplot as plt

layers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
tests=[611, 686, 748, 784, 825, 852, 872, 888, 900, 900, 900]
percentage=[]
for test in tests:
    temp=(test/900)*100
    percentage.append(temp)



plt.figure(figsize=(8, 4))
plt.plot(layers, tests, label='tests', marker='o', linestyle='-')
plt.xlabel('Layers')
plt.ylabel('Tests passed')
plt.title('Logical Tests passed (out of 900) by a handcrafted BERT model')
plt.grid(True)
plt.legend()


# for x, y in zip(layers, tests):
#     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.savefig('plots/tests_vs_layers.png')


plt.figure(figsize=(8, 4))
plt.plot(layers, percentage, label='tests', marker='o', linestyle='-')
plt.xlabel('Layers')
plt.ylabel('Success percentage')
plt.title('Logical Tests passed by a handcrafted BERT model')
plt.grid(True)
#plt.legend()


# for x, y in zip(layers, percentage):
#     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.savefig('plots/tests_vs_layers_percentage.png')


############################################################################################

depth = [0, 1, 2, 3, 4, 5, 6]
lp_on_lp = [0.9991, 0.9987, 0.9990, 0.9972, 0.9947,  0.9899, 0.96814]
lp_on_rp = [0.9958, 0.9562, 0.6857, 0.5959, 0.6140, 0.6602, 0.6573]
rp_on_rp = [0.9977, 0.9928, 0.9774, 0.9797, 0.9762, 0.9555, 0.9534]
rp_on_lp = [0.9958, 0.9562, 0.6857, 0.5959, 0.6140, 0.6602, 0.6573]

plt.figure(figsize=(8, 6))
plt.plot(depth, lp_on_lp, marker='o', label='Evaluated on LP-sampled data')
plt.plot(depth, lp_on_rp, marker='x', label='Evaluated on RP-sampled data')
plt.xlabel('Depth of problems')
plt.ylabel('Success precentage')
plt.title('Efficiency of LP-trained T5 model')
plt.legend()
plt.grid(True)
plt.savefig('plots/lp.png')


plt.figure(figsize=(8, 6))
plt.plot(depth, rp_on_rp, marker='o', label='Evaluated on RP-sampled data')
plt.plot(depth, rp_on_lp, marker='x', label='Evaluated on LP-sampled data')
plt.xlabel('Depth of problems')
plt.ylabel('Success precentage')
plt.title('Efficiency of LP-trained T5 model')
plt.legend()
plt.grid(True)
plt.savefig('plots/rp.png')