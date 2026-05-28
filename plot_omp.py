import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('lab2_results.csv')

sizes = sorted(df['size'].unique())
threads = sorted(df['threads'].unique())

# Создаём два графика
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Время выполнения для разного числа потоков
colors = ['blue', 'green', 'orange', 'red']
for i, t in enumerate(threads):
    data = df[df['threads'] == t]
    ax1.plot(data['size'], data['time_ms'], 'o-', linewidth=2, markersize=6, 
             color=colors[i], label=f'{t} потоков')

ax1.set_xlabel('Размер матрицы N', fontsize=12)
ax1.set_ylabel('Время выполнения (мс)', fontsize=12)
ax1.set_title('Зависимость времени от числа потоков', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Ускорение (Speedup)
for i, n in enumerate(sizes):
    data = df[df['size'] == n].sort_values('threads')
    t1 = data[data['threads'] == 1]['time_ms'].values[0]
    speedup = t1 / data['time_ms'].values
    ax2.plot(data['threads'], speedup, 'o-', linewidth=2, markersize=6,
             label=f'N={n}')

# Идеальное ускорение
ax2.plot(threads, threads, 'k--', linewidth=2, label='Идеальное ускорение')

ax2.set_xlabel('Количество потоков', fontsize=12)
ax2.set_ylabel('Ускорение (Speedup)', fontsize=12)
ax2.set_title('Масштабирование по потокам', fontsize=14)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab2_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ График сохранён как lab2_plot.png")

# Вывод эффективности
print("\n" + "="*70)
print("ЭФФЕКТИВНОСТЬ ПАРАЛЛЕЛИЗАЦИИ (Efficiency = Speedup / P)")
print("="*70)
print(f"{'Размер':>8} | {'Eff 2':>10} | {'Eff 4':>10} | {'Eff 8':>10}")
print("-"*70)

for n in sizes:
    data = df[df['size'] == n].sort_values('threads')
    t1 = data[data['threads'] == 1]['time_ms'].values[0]
    eff2 = (t1 / data[data['threads'] == 2]['time_ms'].values[0]) / 2 if 2 in data['threads'].values else 0
    eff4 = (t1 / data[data['threads'] == 4]['time_ms'].values[0]) / 4 if 4 in data['threads'].values else 0
    eff8 = (t1 / data[data['threads'] == 8]['time_ms'].values[0]) / 8 if 8 in data['threads'].values else 0
    print(f"{n:8d} | {eff2:10.2f} | {eff4:10.2f} | {eff8:10.2f}")
print("="*70)