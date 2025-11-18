import numpy as np
from mpi4py import MPI
import time
import cProfile
import io
import pstats

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000  # Размер матрицы, можно увеличить для тестов

def matrix_vector_multiply():
    if rank == 0:
        A = np.random.rand(N, N)
        b = np.random.rand(N)
    else:
        A = None
        b = None

    # Раздача вектора b всем процессам
    b = comm.bcast(b, root=0)

    # Разделение матрицы по строкам
    rows_per_proc = N // size
    extra_rows = N % size

    if rank == 0:
        local_A = A[:rows_per_proc + extra_rows]
        start = rows_per_proc + extra_rows
        for i in range(1, size):
            end = start + rows_per_proc
            comm.Send(A[start:end], dest=i, tag=0)
            start = end
    else:
        local_rows = rows_per_proc
        local_A = np.empty((local_rows, N))
        comm.Recv(local_A, source=0, tag=0)

    # Локальное умножение
    local_result = np.dot(local_A, b)

    # Сбор результатов
    if rank == 0:
        result = np.empty(N)
        offset = 0
        result[offset:offset + len(local_result)] = local_result
        offset += len(local_result)
        for i in range(1, size):
            recv_buf = np.empty(rows_per_proc)
            comm.Recv(recv_buf, source=i, tag=1)
            result[offset:offset + len(recv_buf)] = recv_buf
            offset += len(recv_buf)
    else:
        comm.Send(local_result, dest=0, tag=1)

    if rank == 0:
        return result

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    result = matrix_vector_multiply()
    end_time = time.time()
    
    pr.disable()
    if rank == 0:
        print(f"Время выполнения: {end_time - start_time} секунд")
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        with open('base_profile.txt', 'w') as f:
            f.write(s.getvalue())