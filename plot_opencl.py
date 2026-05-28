import matplotlib.pyplot as plt
import numpy as np

# Данные
sizes = [200, 400, 800, 1200, 1600, 2000]

# CPU данные (из ЛР №3)
cpu_time = [2.65, 22.02, 182.41, 608.90, 1993.82, 4665.74]
cpu_gflops = [6.04, 5.81, 5.61, 5.68, 4.11, 3.43]

# GPU данные (из ЛР №4)
gpu_time = [6.73, 10.55, 39.63, 60.77, 131.65, 226.90]
gpu_gflops = [2.38, 12.13, 25.84, 56.87, 62.22, 70.52]

# Ускорение GPU относительно CPU
speedup_gpu = [cpu_time[i] / gpu_time[i] for i in range(len(sizes))]

# ============================================
# ГРАФИК 1: Время выполнения (CPU vs GPU)
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, cpu_time, 'o-', linewidth=2, markersize=8, color='blue', label='CPU (MPI)')
plt.plot(sizes, gpu_time, 's-', linewidth=2, markersize=8, color='green', label='GPU (OpenCL)')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Время (мс)', fontsize=12)
plt.title('Сравнение времени выполнения: CPU vs GPU', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_time.png', dpi=150)
plt.show()
print("✓ График 1 сохранён: plot1_time.png")

# ============================================
# ГРАФИК 2: Производительность (GFLOPS)
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, cpu_gflops, 'o-', linewidth=2, markersize=8, color='blue', label='CPU (MPI)')
plt.plot(sizes, gpu_gflops, 's-', linewidth=2, markersize=8, color='green', label='GPU (OpenCL)')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('GFLOPS', fontsize=12)
plt.title('Сравнение производительности: CPU vs GPU', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot2_gflops.png', dpi=150)
plt.show()
print("✓ График 2 сохранён: plot2_gflops.png")

# ============================================
# ГРАФИК 3: Ускорение GPU относительно CPU
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, speedup_gpu, 'd-', linewidth=2, markersize=8, color='red')
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Уровень CPU (1.0)')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('Ускорение (CPU/GPU время)', fontsize=12)
plt.title('Во сколько раз GPU быстрее CPU', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot3_speedup.png', dpi=150)
plt.show()
print("✓ График 3 сохранён: plot3_speedup.png")

# ============================================
# ГРАФИК 4: Производительность GPU
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(sizes, gpu_gflops, 's-', linewidth=2, markersize=8, color='green')
plt.fill_between(sizes, gpu_gflops, alpha=0.3, color='green')
plt.xlabel('Размер матрицы N', fontsize=12)
plt.ylabel('GFLOPS', fontsize=12)
plt.title('Производительность GPU на разных размерах', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot4_gpu_perf.png', dpi=150)
plt.show()
print("✓ График 4 сохранён: plot4_gpu_perf.png")

# ============================================
# Вывод таблицы в консоль
# ============================================
print("\n" + "="*70)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ CPU vs GPU")
print("="*70)
print(f"{'N':>6} | {'CPU время':>12} | {'GPU время':>12} | {'Ускорение':>10} | {'CPU GFLOPS':>10} | {'GPU GFLOPS':>10}")
print("-"*70)
for i, n in enumerate(sizes):
    print(f"{n:6d} | {cpu_time[i]:12.2f} | {gpu_time[i]:12.2f} | {speedup_gpu[i]:10.2f}x | {cpu_gflops[i]:10.2f} | {gpu_gflops[i]:10.2f}")
print("="*70)

print("\n📊 АНАЛИЗ:")
print(f"• Максимальное ускорение GPU: {max(speedup_gpu):.2f}x (при N={sizes[speedup_gpu.index(max(speedup_gpu))]})")
print(f"• Максимальная производительность GPU: {max(gpu_gflops):.2f} GFLOPS")
print(f"• Максимальная производительность CPU: {max(cpu_gflops):.2f} GFLOPS")
print(f"• GPU начинает превосходить CPU при N={sizes[next(i for i, s in enumerate(speedup_gpu) if s > 1)]}")