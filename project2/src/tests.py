import subprocess
import os

# image paths
MAT5 = "./matrices/matrix5.txt"
MAT6 = "./matrices/matrix6.txt"
REF_RESULT = "./build/ref_result.txt"
RESULT = "./build/result.txt"

# programs
naive = "srun -n 1 --cpus-per-task 1 ./build/src/naive"
locality = "srun -n 1 --cpus-per-task 1 ./build/src/locality"
simd = "srun -n 1 --cpus-per-task 1 ./build/src/simd"
simd_aligned = "srun -n 1 --cpus-per-task 1 ./build/src/simd_aligned"


def build():
    """run build command."""
    subprocess.run(["cmake", "-B", "build"])
    subprocess.run(["cmake", "--build", "build", "--parallel", "4"])


def create_ref_result():
    if not os.path.exists(REF_RESULT):
        print("creating reference result")
        subprocess.run(run_mat_command(naive, REF_RESULT))


def run_test(program: str, *extra_args):
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    name = program.split("/")[-1]
    print(f"{HEADER}==== Testing {name} ===={ENDC}")
    subprocess.run(run_mat_command(program, *extra_args))
    if compare_files(REF_RESULT, RESULT):
        print(f"{OKGREEN}Test {name} passed{ENDC}\n")
    else:
        print(f"{FAIL}Test {name} failed!{ENDC}\n")


def run_mat_command(program: str, *extra_args):
    """the command to run program with lena.jpg as test image."""
    cmd = program.split(" ")
    cmd +=[MAT5, MAT6]
    cmd += [str(s) for s in extra_args]
    return cmd


def compare_files(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        byte1 = f1.read()
        byte2 = f2.read()
        return byte1 == byte2


if __name__ == "__main__":
    build()
    create_ref_result()
    run_test(locality, RESULT)
    run_test(simd, RESULT)
    run_test(simd_aligned, RESULT)
