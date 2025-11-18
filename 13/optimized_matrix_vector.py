import numpy as np
from mpi4py import MPI
import time
import cProfile
import io
import pstats

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000  # Увеличьте для тестов, чтобы увидеть масштабируемость

def optimized_matrix_vector_multiply():
    if rank == 0:
        A = np.random.rand(N, N)
        b = np.random.rand(N)
    else:
        A = None
        b = None

    # Broadcast вектора b (малый объем, синхронный)
    b = comm.bcast(b, root=0)

    # Вычисление размеров локальных частей (list для простоты)
    rows_per_proc = [N // size] * size
    extra = N % size
    for i in range(extra):
        rows_per_proc[i] += 1

    # Для Scatterv: counts и displs как np.array (элементы, не строки)
    sendcounts = np.array(rows_per_proc) * N
    send_displs = np.cumsum([0] + rows_per_proc[:-1]) * N

    # Локальный буфер (flat)
    local_rows = rows_per_proc[rank]
    local_A = np.empty(local_rows * N)

    # Sendbuf только в root
    sendbuf = [A.flatten(), sendcounts, send_displs, MPI.DOUBLE] if rank == 0 else None

    # Scatterv
    comm.Scatterv(sendbuf, local_A, root=0)

    # Reshape
    local_A = local_A.reshape((local_rows, N))

    # Локальное вычисление
    local_result = np.dot(local_A, b)

    # Для Gatherv: recvcounts и displs как np.array (для вектора результата)
    recvcounts = np.array(rows_per_proc, dtype='i4')  # Число элементов от каждого proc
    recv_displs = np.cumsum([0] + rows_per_proc[:-1], dtype='i4')  # Смещения

    if rank == 0:
        result = np.empty(N)
    else:
        result = None

    # Recvbuf только в root
    recvbuf = [result, recvcounts, recv_displs, MPI.DOUBLE] if rank == 0 else None

    # Gatherv
    comm.Gatherv(local_result, recvbuf, root=0)

    # Декартова топология (как в лекции стр. 2-3 для оптимизации обменов, здесь 1D)
    dims = MPI.Compute_dims(size, 1)
    cart_comm = comm.Create_cart(dims, periods=False)

    if rank == 0:
        return result

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    result = optimized_matrix_vector_multiply()
    end_time = time.time()
    
    pr.disable()
    if rank == 0:
        print(f"Оптимизированное время: {end_time - start_time} секунд")
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        with open('optimized_profile.txt', 'w') as f:
            f.write(s.getvalue())