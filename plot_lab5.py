import matplotlib.pyplot as plt
import numpy as np

# Данные из результатов
sizes = [200, 400, 800, 1200, 1600, 2000]

# Время выполнения (секунды) для разного числа процессов
time_1 = [0.159365, 1.32732, 10.0016, 34.3703, 83.2583, 177.558]
time_2 = [0.0425699, 0.313009, 2.3806, 7.69469, 17.5219, 33.4421]
time_4 = [0.023133, 0.162481, 1.25673, 3.7333, 8.70601, 16.9141]
time_8 = [0.014595, 0.0878639, 0.650489, 2.05687, 4.61159, 8.92465]
time_16 = [0.0183899, 0.104688, 0.706818, 2.1518, 4.82181, 9.26008]

# Конвертируем в миллисекунды для удобства
time_1_ms = [t * 1000 for t in time_1]
time_2_ms = [t * 1000 for t in time_2]
time_4_ms = [t * 1000 for t in time_4]
time_8_ms = [t * 1000 for t in time_8]
time_16_ms = [t * 1000 for t in time_16]

# Вычисляем производительность (GFLOPS)
# ops = 2 * n^3
def calc_gflops(t_sec, n):
    ops = 2 * n * n * n
    return ops / t_sec / 1e9

gflops_1 = [calc_gflops(time_1[i], sizes[i]) for i in range(len(sizes))]
gflops_2 = [calc_gflops(time_2[i], sizes[i]) for i in range(len(sizes))]
gflops_4 = [calc_gflops(time_4[i], sizes[i]) for i in range(len(sizes))]
gflops_8 = [calc_gflops(time_8[i], sizes[i]) for i in range(len(sizes))]
gflops_16 = [calc_gflops(time_16[i], sizes[i]) for i in range(len(sizes))]

# Вычисляем ускорение относительно 1 процесса
speedup_2 = [time_1[i] / time_2[i] for i in range(len(sizes))]
speedup_4 = [time_1[i] / time_4[i] for i in range(len(sizes))]
speedup_8 = [time_1[i] / time_8[i] for i in range(len(sizes))]
speedup_16 = [time_1[i] / time_16[i] for i in range(len(sizes))]

# ============================================
# ГРАФИК 1: Время выполнения
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, time_1_ms, 'o-', linewidth=2, markersize=8, color='blue', label='1 процесс')
plt.plot(sizes, time_2_ms, 's-', linewidth=2, markersize=8, color='green', label='2 процесса')
plt.plot(sizes, time_4_ms, '^-', linewidth=2, markersize=8, color='orange', label='4 процесса')
plt.plot(sizes, time_8_ms, 'd-', linewidth=2, markersize=8, color='red', label='8 процессов')
plt.plot(sizes, time_16_ms, '*-', linewidth=2, markersize=8, color='purple', label='16 процессов')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время (мс)', fontsize=12)
plt.title('MPI на суперкомпьютере: время выполнения', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_plot1_time.png', dpi=150)
plt.show()
print("✓ График 1 сохранён: lab5_plot1_time.png")

# ============================================
# ГРАФИК 2: Производительность (GFLOPS)
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, gflops_1, 'o-', linewidth=2, markersize=8, color='blue', label='1 процесс')
plt.plot(sizes, gflops_2, 's-', linewidth=2, markersize=8, color='green', label='2 процесса')
plt.plot(sizes, gflops_4, '^-', linewidth=2, markersize=8, color='orange', label='4 процесса')
plt.plot(sizes, gflops_8, 'd-', linewidth=2, markersize=8, color='red', label='8 процессов')
plt.plot(sizes, gflops_16, '*-', linewidth=2, markersize=8, color='purple', label='16 процессов')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('GFLOPS', fontsize=12)
plt.title('MPI на суперкомпьютере: производительность', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_plot2_gflops.png', dpi=150)
plt.show()
print("✓ График 2 сохранён: lab5_plot2_gflops.png")

# ============================================
# ГРАФИК 3: Ускорение
# ============================================
plt.figure(figsize=(10, 6))
ideal = [1, 2, 4, 8, 16]
plt.plot(sizes, speedup_2, 's-', linewidth=2, markersize=8, color='green', label='2 / 1')
plt.plot(sizes, speedup_4, '^-', linewidth=2, markersize=8, color='orange', label='4 / 1')
plt.plot(sizes, speedup_8, 'd-', linewidth=2, markersize=8, color='red', label='8 / 1')
plt.plot(sizes, speedup_16, '*-', linewidth=2, markersize=8, color='purple', label='16 / 1')
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Ускорение (Speedup)', fontsize=12)
plt.title('MPI на суперкомпьютере: ускорение', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_plot3_speedup.png', dpi=150)
plt.show()
print("✓ График 3 сохранён: lab5_plot3_speedup.png")

# ============================================
# ГРАФИК 4: Сравнение ускорения с идеальным (для N=2000)
# ============================================
processes = [1, 2, 4, 8, 16]
speedup_2000 = [1.0, speedup_2[5], speedup_4[5], speedup_8[5], speedup_16[5]]
ideal_speedup = [1, 2, 4, 8, 16]

plt.figure(figsize=(10, 6))
plt.plot(processes, speedup_2000, 'o-', linewidth=2, markersize=8, color='blue', label='Реальное ускорение (N=2000)')
plt.plot(processes, ideal_speedup, 'r--', linewidth=2, markersize=8, color='red', label='Идеальное ускорение')
plt.xlabel('Количество процессов', fontsize=12)
plt.ylabel('Ускорение (Speedup)', fontsize=12)
plt.title('Сравнение реального и идеального ускорения (матрица 2000×2000)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_plot4_comparison.png', dpi=150)
plt.show()
print("✓ График 4 сохранён: lab5_plot4_comparison.png")

# ============================================
# Вывод таблицы в консоль
# ============================================
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ MPI НА СУПЕРКОМПЬЮТЕРЕ")
print("="*80)

print("\n【Время выполнения (мс)】")
print(f"{'N':>6} | {'1':>10} | {'2':>10} | {'4':>10} | {'8':>10} | {'16':>10}")
print("-"*70)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {time_1_ms[i]:10.2f} | {time_2_ms[i]:10.2f} | {time_4_ms[i]:10.2f} | {time_8_ms[i]:10.2f} | {time_16_ms[i]:10.2f}")

print("\n【Производительность (GFLOPS)】")
print(f"{'N':>6} | {'1':>10} | {'2':>10} | {'4':>10} | {'8':>10} | {'16':>10}")
print("-"*70)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {gflops_1[i]:10.2f} | {gflops_2[i]:10.2f} | {gflops_4[i]:10.2f} | {gflops_8[i]:10.2f} | {gflops_16[i]:10.2f}")

print("\n【Ускорение (Speedup)】")
print(f"{'N':>6} | {'2/1':>10} | {'4/1':>10} | {'8/1':>10} | {'16/1':>10}")
print("-"*70)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {speedup_2[i]:10.2f}x | {speedup_4[i]:10.2f}x | {speedup_8[i]:10.2f}x | {speedup_16[i]:10.2f}x")

print("\n" + "="*80)
print("📊 АНАЛИЗ:")
print(f"• Лучшее ускорение (16 процессов / 1 процесс): {max(speedup_16):.2f}x (при N={sizes[speedup_16.index(max(speedup_16))]})")
print(f"• Лучшая производительность: {max(gflops_16):.2f} GFLOPS (16 процессов, N=2000)")
print(f"• Эффективность на 16 процессах (N=2000): {speedup_16[5]/16*100:.1f}%")
print("="*80)