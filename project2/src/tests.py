import subprocess
import os

HEADER = '\033[95m'
OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

# image paths
MAT = "./matrices/matrix{}.txt"
REF_RESULT = "./build/ref_result_{}.txt"
RESULT = "./build/result.txt"
PERF_OUT_FOLDER = "./profiling"

TESTS_SUITS = [
    (REF_RESULT.format(12), MAT.format(1), MAT.format(2)),
    (REF_RESULT.format(34), MAT.format(3), MAT.format(4)),
    (REF_RESULT.format(56), MAT.format(5), MAT.format(6)),
    (REF_RESULT.format(78), MAT.format(7), MAT.format(8)),
]

# programs
naive_old = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/naive_old"
naive = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/naive"
locality = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/locality"
simd = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/simd"
simd_aligned = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/simd_aligned"
simd_tiled = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/simd_tiled"
simd_aligned_tiled = "srun -n 1 --cpus-per-task 1 --partition Project ./build/src/simd_aligned_tiled"
openmp = "srun -n 1 --cpus-per-task {ncpu} --partition Project ./build/src/openmp {ncpu}"
mpi = "srun -n {nproc} --cpus-per-task {ncpu} --partition Project --mpi=pmi2 ./build/src/mpi {ncpu}"

basic_perf_naive_old = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/naive_old"
basic_perf_naive = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/naive"
basic_perf_locality = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/locality"
basic_perf_simd = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/simd"
basic_perf_simd_aligned = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/simd_aligned"
basic_perf_simd_tiled = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/simd_tiled"
basic_perf_simd_aligned_tiled = "srun -n 1 --cpus-per-task 1 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/simd_aligned_tiled"
basic_perf_openmp = "srun -n 1 --cpus-per-task {ncpu} --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/openmp {ncpu}"
basic_perf_mpi = "srun -n {nproc} --cpus-per-task {ncpu} --mpi=pmi2 --partition Project perf stat -e cpu-cycles,cache-misses,page-faults,branch-misses ./build/src/mpi {ncpu}"

detailed_perf_naive_old = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/naive_old.data ./build/src/naive_old"
detailed_perf_naive = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/naive.data ./build/src/naive"
detailed_perf_locality = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/locality.data ./build/src/locality"
detailed_perf_simd = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/simd.data ./build/src/simd"
detailed_perf_simd_aligned = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/simd_aligned.data ./build/src/simd_aligned"
detailed_perf_simd_tiled = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/simd_tiled.data ./build/src/simd_tiled"
detailed_perf_simd_aligned_tiled = "srun -n 1 --cpus-per-task 1 --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/simd_aligned_tiled.data ./build/src/simd_aligned_tiled"
detailed_perf_openmp = "srun -n 1 --cpus-per-task {ncpu} --partition Project perf record -e cpu-cycles:pp,cache-misses,page-faults,branch-misses -g -o ./profiling/openmp-{ncpu}.data ./build/src/openmp {ncpu}"

def build():
    """run build command."""
    subprocess.run(["cmake", "-B", "build"])
    subprocess.run(["cmake", "--build", "build", "--parallel", "4"])


def create_ref_results():
    for ref_result, mat_a, mat_b in TESTS_SUITS:
        if not os.path.exists(ref_result):
            print(f"creating reference result {ref_result}")
            subprocess.run(run_mat_command(naive, mat_a, mat_b, ref_result))


def run_test(program: str, ref_result, mats, *extra_args):
    name = program.split("/")[-1]
    print(f"{HEADER}==== Testing {name} ===={ENDC}")
    subprocess.run(run_mat_command(program, *mats, RESULT, *extra_args))
    if compare_files(ref_result, RESULT):
        print(f"{OKGREEN}Test {name} passed{ENDC}\n")
    else:
        print(f"{FAIL}Test {name} failed!{ENDC}\n")


def run_mat_command(program: str, *extra_args):
    """the command to run program with lena.jpg as test image."""
    cmd = program.split(" ")
    cmd += [str(s) for s in extra_args]
    return cmd


def compare_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        byte1 = f1.read()
        byte2 = f2.read()
        return byte1 == byte2


def tests_with_data(ref_result, mats):
    run_test(naive_old, ref_result, mats)
    run_test(naive, ref_result, mats)
    run_test(locality, ref_result, mats)
    run_test(simd, ref_result, mats)
    run_test(simd_aligned, ref_result, mats)
    run_test(simd_tiled, ref_result, mats)
    run_test(simd_aligned_tiled, ref_result, mats)
    run_test(openmp.format(ncpu=1), ref_result, mats)
    run_test(openmp.format(ncpu=4), ref_result, mats)
    run_test(openmp.format(ncpu=32), ref_result, mats)
    run_test(mpi.format(nproc=1, ncpu=32), ref_result, mats)
    run_test(mpi.format(nproc=4, ncpu=8), ref_result, mats)
    run_test(mpi.format(nproc=8, ncpu=4), ref_result, mats)
    run_test(mpi.format(nproc=32, ncpu=1), ref_result, mats)


def run_perf(ref_result, mats):
    run_test(basic_perf_naive_old, ref_result, mats)
    run_test(basic_perf_naive, ref_result, mats)
    run_test(basic_perf_locality, ref_result, mats)
    run_test(basic_perf_simd, ref_result, mats)
    run_test(basic_perf_simd_aligned, ref_result, mats)
    run_test(basic_perf_simd_tiled, ref_result, mats)
    run_test(basic_perf_simd_aligned_tiled, ref_result, mats)
    run_test(basic_perf_openmp.format(ncpu=1), ref_result, mats)
    run_test(basic_perf_openmp.format(ncpu=4), ref_result, mats)
    run_test(basic_perf_openmp.format(ncpu=32), ref_result, mats)
    run_test(basic_perf_mpi.format(nproc=1, ncpu=32), ref_result, mats)
    run_test(basic_perf_mpi.format(nproc=4, ncpu=8), ref_result, mats)
    run_test(basic_perf_mpi.format(nproc=8, ncpu=4), ref_result, mats)
    run_test(basic_perf_mpi.format(nproc=32, ncpu=1), ref_result, mats)
    if not os.path.exists(PERF_OUT_FOLDER):
        os.makedirs(PERF_OUT_FOLDER)
    run_test(detailed_perf_naive_old, ref_result, mats)
    run_test(detailed_perf_naive, ref_result, mats)
    run_test(detailed_perf_locality, ref_result, mats)
    run_test(detailed_perf_simd, ref_result, mats)
    run_test(detailed_perf_simd_aligned, ref_result, mats)
    run_test(detailed_perf_simd_tiled, ref_result, mats)
    run_test(detailed_perf_simd_aligned_tiled, ref_result, mats)
    run_test(detailed_perf_openmp.format(ncpu=1), ref_result, mats)
    run_test(detailed_perf_openmp.format(ncpu=4), ref_result, mats)
    run_test(detailed_perf_openmp.format(ncpu=32), ref_result, mats)


if __name__ == "__main__":
    build()
    create_ref_results()
    for ref_result, mat_a, mat_b in TESTS_SUITS:
        print(f"{HEADER}>>>==== TESTING WITH MATS {[mat_a, mat_b]} ====<<<{ENDC}")
        tests_with_data(ref_result, [mat_a, mat_b])
    print(f"{HEADER}>>>==== Start Running Basic Perf ====<<<{ENDC}")
    ref_result, mat_a, mat_b = TESTS_SUITS[3]
    run_perf(ref_result, [mat_a, mat_b])
