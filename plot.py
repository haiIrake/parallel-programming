import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Читаем результаты
df = pd.read_csv('lab1_results.csv')
sizes = df['size'].values
times = df['time_ms'].values
gflops = df['gflops'].values

# Теоретическая кривая O(N³) — масштабирование от первого измерения
theoretical = times[0] * (sizes / sizes[0]) ** 3

# Создаём два графика
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ========== ГРАФИК 1: Время выполнения ==========
ax1.plot(sizes, times, 'o-', linewidth=2.5, markersize=8, color='navy', 
         label='Эксперимент')
ax1.plot(sizes, theoretical, 'r--', linewidth=2.5, 
         label='Теория O(N³)')
ax1.set_xlabel('Размер матрицы N', fontsize=12)
ax1.set_ylabel('Время выполнения (мс)', fontsize=12)
ax1.set_title('Зависимость времени от размера матрицы', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# ========== ГРАФИК 2: Производительность ==========
ax2.plot(sizes, gflops, 's-', linewidth=2.5, markersize=8, color='darkgreen',
         label='Производительность')
ax2.axhline(y=np.mean(gflops), color='orange', linestyle=':', linewidth=2,
            label=f'Среднее = {np.mean(gflops):.2f} GFLOPS')
ax2.set_xlabel('Размер матрицы N', fontsize=12)
ax2.set_ylabel('Производительность (GFLOPS)', fontsize=12)
ax2.set_title('Эффективность алгоритма', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab1_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# ========== ВЫВОД ТАБЛИЦЫ ==========
print("\n" + "="*65)
print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
print("="*65)
print(f"{'N':>6} | {'Время (мс)':>12} | {'Теория (мс)':>12} | {'GFLOPS':>8}")
print("-"*65)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {times[i]:12.2f} | {theoretical[i]:12.2f} | {gflops[i]:8.2f}")
print("="*65)

# Коэффициент масштабирования
print(f"\nМасштабирование 200 → 2000:")
print(f"  Теория: 1000x")
print(f"  Эксперимент: {times[-1]/times[0]:.1f}x")
print(f"\n✓ График сохранён как 'lab1_plot.png'")