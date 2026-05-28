import subprocess
import numpy as np
import struct
import json
import os
import time
import csv

def write_binary_matrix(filename, matrix):
    n = matrix.shape[0]
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', n))
        f.write(matrix.astype(np.float64).tobytes())

def run_multiplication(n, num_threads):
    print(f"  Потоков: {num_threads}")
    
    np.random.seed(42)
    A = np.random.randn(n, n) * 10
    B = np.random.randn(n, n) * 10
    
    write_binary_matrix(f"temp_A.bin", A)
    write_binary_matrix(f"temp_B.bin", B)
    
    json_file = f"result_{n}_{num_threads}.json"
    
    start = time.time()
    result = subprocess.run(
        ["./matmul_omp.exe", "temp_A.bin", "temp_B.bin", "temp_C.bin", str(num_threads), json_file],
        capture_output=True, text=True
    )
    elapsed = time.time() - start
    
    if result.returncode == 0 and os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data['time_ms'], data['gflops']
    else:
        print(f"    Ошибка: {result.stderr}")
        return None, None

def main():
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: OpenMP")
    print("Умножение матриц с параллелизацией")
    print("="*70)
    
    sizes = [200, 400, 800, 1200, 1600, 2000]
    thread_counts = [1, 2, 4, 8]
    
    all_results = []
    
    for n in sizes:
        print(f"\n{'='*60}")
        print(f"РАЗМЕР МАТРИЦЫ: {n} x {n}")
        print(f"{'='*60}")
        
        for threads in thread_counts:
            time_ms, gflops = run_multiplication(n, threads)
            
            if time_ms:
                print(f"    Время: {time_ms:.2f} мс, GFLOPS: {gflops:.2f}")
                all_results.append({
                    "size": n,
                    "threads": threads,
                    "time_ms": time_ms,
                    "gflops": gflops
                })
        
        print()
    
    # Сохраняем результаты
    with open('lab2_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'threads', 'time_ms', 'gflops'])
        writer.writeheader()
        writer.writerows(all_results)
    
    # Вывод итоговой таблицы
    print("\n\n" + "="*80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*80)
    print(f"{'Размер':>8} | {'1 поток':>12} | {'2 потока':>12} | {'4 потока':>12} | {'8 потоков':>12}")
    print("-"*80)
    
    for n in sizes:
        row = [r for r in all_results if r['size'] == n]
        times = {r['threads']: r['time_ms'] for r in row}
        print(f"{n:8d} | {times.get(1, 0):12.2f} | {times.get(2, 0):12.2f} | {times.get(4, 0):12.2f} | {times.get(8, 0):12.2f}")
    
    print("="*80)
    
    # Расчёт ускорения
    print("\n" + "="*80)
    print("УСКОРЕНИЕ (Speedup)")
    print("="*80)
    print(f"{'Размер':>8} | {'2/1':>10} | {'4/1':>10} | {'8/1':>10}")
    print("-"*80)
    
    for n in sizes:
        row = [r for r in all_results if r['size'] == n]
        times = {r['threads']: r['time_ms'] for r in row}
        if 1 in times:
            t1 = times[1]
            s2 = t1 / times.get(2, t1) if 2 in times else 0
            s4 = t1 / times.get(4, t1) if 4 in times else 0
            s8 = t1 / times.get(8, t1) if 8 in times else 0
            print(f"{n:8d} | {s2:10.2f}x | {s4:10.2f}x | {s8:10.2f}x")
    
    print("="*80)
    print("\n✓ Результаты сохранены в lab2_results.csv")

if __name__ == "__main__":
    main()