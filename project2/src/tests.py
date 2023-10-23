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

TESTS_SUITS = [
    (REF_RESULT.format(12), MAT.format(1), MAT.format(2)),
    (REF_RESULT.format(34), MAT.format(3), MAT.format(4)),
    (REF_RESULT.format(56), MAT.format(5), MAT.format(6)),
    (REF_RESULT.format(78), MAT.format(7), MAT.format(8)),
]

# programs
naive = "srun -n 1 --cpus-per-task 1 ./build/src/naive"
locality = "srun -n 1 --cpus-per-task 1 ./build/src/locality"
simd = "srun -n 1 --cpus-per-task 1 ./build/src/simd"
simd_aligned = "srun -n 1 --cpus-per-task 1 ./build/src/simd_aligned"
simd_tiled = "srun -n 1 --cpus-per-task 1 ./build/src/simd_tiled"
simd_aligned_tiled = "srun -n 1 --cpus-per-task 1 ./build/src/simd_aligned_tiled"
openmp = "srun -n 1 --cpus-per-task {ncpu} ./build/src/openmp {ncpu}"


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
    run_test(locality, ref_result, mats)
    run_test(simd, ref_result, mats)
    run_test(simd_aligned, ref_result, mats)
    run_test(simd_tiled, ref_result, mats)
    run_test(simd_aligned_tiled, ref_result, mats)
    run_test(openmp.format(ncpu=1), ref_result, mats)
    run_test(openmp.format(ncpu=4), ref_result, mats)
    run_test(openmp.format(ncpu=32), ref_result, mats)


if __name__ == "__main__":
    build()
    create_ref_results()
    for ref_result, mat_a, mat_b in TESTS_SUITS:
        print(f"{HEADER}>>>==== TESTING WITH MATS {[mat_a, mat_b]} ====<<<{ENDC}")
        tests_with_data(ref_result, [mat_a, mat_b])
