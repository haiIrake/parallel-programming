import matplotlib.pyplot as plt
import numpy as np

# Данные из эксперимента (вставлены прямо в код)
data = [
    (200, 1, 2.65, 6.04),
    (200, 2, 1.20, 13.29),
    (200, 4, 0.65, 24.56),
    (200, 8, 0.37, 43.61),
    (400, 1, 22.02, 5.81),
    (400, 2, 11.37, 11.25),
    (400, 4, 5.82, 21.98),
    (400, 8, 2.66, 48.09),
    (800, 1, 182.41, 5.61),
    (800, 2, 100.77, 10.16),
    (800, 4, 54.91, 18.65),
    (800, 8, 37.95, 26.98),
    (1200, 1, 608.90, 5.68),
    (1200, 2, 402.67, 8.58),
    (1200, 4, 254.44, 13.58),
    (1200, 8, 223.84, 15.44),
    (1600, 1, 1993.82, 4.11),
    (1600, 2, 1029.56, 7.96),
    (1600, 4, 718.69, 11.40),
    (1600, 8, 579.19, 14.14),
    (2000, 1, 4665.74, 3.43),
    (2000, 2, 2506.18, 6.38),
    (2000, 4, 1538.38, 10.40),
    (2000, 8, 1150.66, 13.91),
]

# Группируем по размеру матрицы
sizes = sorted(set(d[0] for d in data))
procs = sorted(set(d[1] for d in data))

# Словари для времени и производительности
time_data = {n: {} for n in sizes}
gflops_data = {n: {} for n in sizes}
for n, p, t, g in data:
    time_data[n][p] = t
    gflops_data[n][p] = g

# Построение графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Время выполнения
colors = ['blue', 'green', 'orange', 'red']
markers = ['o', 's', '^', 'd']
for i, p in enumerate(procs):
    times = [time_data[n][p] for n in sizes]
    ax1.plot(sizes, times, marker=markers[i], linestyle='-', linewidth=2,
             markersize=8, color=colors[i], label=f'{p} процессов')
ax1.set_xlabel('Размер матрицы N', fontsize=12)
ax1.set_ylabel('Время (мс)', fontsize=12)
ax1.set_title('Время выполнения MPI-умножения', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Ускорение относительно 1 процесса
for n in sizes:
    t1 = time_data[n][1]
    speedup = [t1 / time_data[n][p] for p in procs]
    ax2.plot(procs, speedup, marker='o', linestyle='-', linewidth=2,
             markersize=8, label=f'N={n}')
ax2.plot(procs, procs, 'k--', linewidth=2, label='Идеальное ускорение')
ax2.set_xlabel('Количество процессов', fontsize=12)
ax2.set_ylabel('Ускорение (Speedup)', fontsize=12)
ax2.set_title('Масштабирование MPI', fontsize=14)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab3_plot.png', dpi=150)
plt.show()

print("График сохранён как lab3_plot.png")