import matplotlib.pyplot as plt
import numpy as np

vlogfreq1 = np.load('vlogfreq1.npy')
vlogfreq2 = np.load('vlogfreq2.npy')
vlogpot1 = np.load('vlogpot1.npy')
vlogpot2 = np.load('vlogpot2.npy')

fig = plt.figure()
plt.plot(vlogfreq1, vlogpot1, 'o', color = 'green', label='Ions trapped')
plt.plot(vlogfreq2, vlogpot2, 'o', color = 'red', label='Ions go away')
plt.title('Ions behaviour depending on potential and frequency')
plt.xlabel(r'$log_{10}(f(Hz))$')
plt.ylabel('Potential (V)')
plt.legend()
plt.show()