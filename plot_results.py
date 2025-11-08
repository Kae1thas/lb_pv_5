import numpy as np
import matplotlib.pyplot as plt

# Твои результаты (замени на свои замеры времени)
procs = [1, 4, 9, 16]
times_1d = [10.2, 4.5, 2.8, 2.1]  # Из ЛР3 (одномерная декомп)
times_2d = [10.2, 3.2, 1.9, 1.5]  # Твои 2D (замерь)

T1 = times_2d[0]
speedup_2d = [T1 / t for t in times_2d]
speedup_1d = [T1 / t for t in times_1d]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(procs, speedup_1d, 'o-', label='1D декомп')
plt.plot(procs, speedup_2d, 's-', label='2D декомп')
plt.title('Ускорение')
plt.xlabel('Процессов')
plt.ylabel('S(p)')
plt.legend()

plt.subplot(1, 2, 2)
efficiency_2d = [s / p for s, p in zip(speedup_2d, procs)]
efficiency_1d = [s / p for s, p in zip(speedup_1d, procs)]
plt.plot(procs, efficiency_1d, 'o-', label='1D')
plt.plot(procs, efficiency_2d, 's-', label='2D')
plt.title('Эффективность')
plt.xlabel('Процессов')
plt.ylabel('E(p)')
plt.legend()
plt.tight_layout()
plt.savefig('lab5_results.png')
plt.show()