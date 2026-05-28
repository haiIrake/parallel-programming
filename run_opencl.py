import subprocess
import re
import csv

# Размеры матриц
sizes = [200, 400, 800, 1200, 1600, 2000]

print("="*50)
print("OPENCL MATRIX MULTIPLICATION")
print("="*50)
print(f"{'Size':>8} | {'Time (ms)':>12} | {'GFLOPS':>10}")
print("-"*50)

results = []

for n in sizes:
    # Запускаем программу
    result = subprocess.run(
        [f"./matmul_opencl", str(n)],
        capture_output=True,
        text=True
    )
    
    # Парсим вывод
    output = result.stdout
    time_match = re.search(r"Time:\s+([\d.]+)", output)
    gflops_match = re.search(r"GFLOPS:\s+([\d.]+)", output)
    
    if time_match and gflops_match:
        time_ms = float(time_match.group(1))
        gflops = float(gflops_match.group(1))
        results.append({
            "size": n,
            "time_ms": time_ms,
            "gflops": gflops
        })
        print(f"{n:8d} | {time_ms:12.2f} | {gflops:10.2f}")
    else:
        print(f"{n:8d} | {'ERROR':12} | {'ERROR':10}")

print("="*50)

# Сохраняем в CSV
with open('lab4_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['size', 'time_ms', 'gflops'])
    writer.writeheader()
    writer.writerows(results)

print("\n✓ Результаты сохранены в lab4_results.csv")