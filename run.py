import subprocess
import numpy as np
import struct
import json
import os
import time

def write_binary_matrix(filename, matrix):
    """Запись матрицы в бинарный файл"""
    n = matrix.shape[0]
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', n))
        f.write(matrix.astype(np.float64).tobytes())

def run_multiplication(n, block_size):
    """Запуск умножения для заданного размера"""
    print(f"\n  Размер блока: {block_size}")
    
    # Генерация случайных матриц
    np.random.seed(42)
    A = np.random.randn(n, n) * 10
    B = np.random.randn(n, n) * 10
    
    write_binary_matrix(f"temp_A.bin", A)
    write_binary_matrix(f"temp_B.bin", B)
    
    # Запуск программы
    json_file = f"result_{n}_{block_size}.json"
    start = time.time()
    
    result = subprocess.run(
        ["./matmul_seq.exe", "temp_A.bin", "temp_B.bin", "temp_C.bin", str(block_size), json_file],
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
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Умножение квадратных матриц (блочный алгоритм)")
    print("="*60)
    
    sizes = [200, 400, 800, 1200, 1600, 2000]
    block_sizes = [32, 64, 128]
    
    results = []
    
    for n in sizes:
        print(f"\n{'='*50}")
        print(f"РАЗМЕР МАТРИЦЫ: {n} x {n}")
        print(f"{'='*50}")
        
        best_time = float('inf')
        best_block = 64
        best_gflops = 0
        
        for bs in block_sizes:
            time_ms, gflops = run_multiplication(n, bs)
            
            if time_ms and time_ms < best_time:
                best_time = time_ms
                best_block = bs
                best_gflops = gflops
            
            if time_ms:
                print(f"    Время: {time_ms:.2f} мс, GFLOPS: {gflops:.2f}")
        
        results.append({
            "size": n,
            "time_ms": best_time,
            "best_block": best_block,
            "gflops": best_gflops
        })
        
        print(f"\n  ★ Лучший: блок={best_block}, время={best_time:.2f} мс")
    
    # Вывод таблицы
    print("\n\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"{'Размер':>8} | {'Время (мс)':>12} | {'Блок':>6} | {'GFLOPS':>10}")
    print("-"*50)
    
    for r in results:
        print(f"{r['size']:8d} | {r['time_ms']:12.2f} | {r['best_block']:6d} | {r['gflops']:10.2f}")
    
    print("="*70)
    
    # Сохранение в CSV
    import csv
    with open('lab1_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'time_ms', 'best_block', 'gflops'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n✓ Результаты сохранены в lab1_results.csv")

if __name__ == "__main__":
    main()