import subprocess
import matplotlib.pyplot as plt

procs = [1, 4, 9, 16]
times_base = [0.0144, 0.033, 0.1391, 0.2159]
times_opt = [0.0285, 0.1287, 0.0835, 0.2987]

T_seq = times_base[0]
speedup_base = [T_seq / t for t in times_base]
efficiency_base = [s / p for s, p in zip(speedup_base, procs)]

speedup_opt = [T_seq / t for t in times_opt]
efficiency_opt = [s / p for s, p in zip(speedup_opt, procs)]

plt.figure()
plt.plot(procs, times_base, label='Базовая')
plt.plot(procs, times_opt, label='Оптимизированная')
plt.xlabel('Число процессов')
plt.ylabel('Время (с)')
plt.legend()
plt.savefig('times.png')

plt.figure()
plt.plot(procs, speedup_base, label='Базовая speedup')
plt.plot(procs, speedup_opt, label='Оптимизированная speedup')
plt.xlabel('Число процессов')
plt.ylabel('Ускорение')
plt.legend()
plt.savefig('speedup.png')

plt.figure()
plt.plot(procs, efficiency_base, label='Базовая efficiency')
plt.plot(procs, efficiency_opt, label='Оптимизированная efficiency')
plt.xlabel('Число процессов')
plt.ylabel('Эффективность')
plt.legend()
plt.savefig('efficiency.png')