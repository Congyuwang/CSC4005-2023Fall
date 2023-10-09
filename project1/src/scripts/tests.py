import subprocess
from typing import List
from PIL import Image
import numpy as np

# test tolerance 15
TOL = 15

# image paths
LENA_IN = "./images/Lena-RGB.jpg"
LENA_REF = "./images/Lena-Smooth.jpg"
LENA_OUT = "./images/Lena-test.jpg"

# programs
sequential_PartB = "./build/src/cpu/sequential_PartB"
simd_PartB = "./build/src/cpu/simd_PartB"
pthread_PartB = "./build/src/cpu/pthread_PartB"
pthread_simd_PartB = "./build/src/cpu/pthread_simd_PartB"
openmp_PartB = "./build/src/cpu/openmp_PartB"
cuda_PartB = "./build/src/gpu/cuda_PartB"
openacc_PartB = "./build/src/gpu/openacc_PartB"
mpi_PartB = "srun -n 4 --cpus-per-task 1 --mpi=pmi2 ./build/src/cpu/mpi_PartB"


def build():
    """run build command."""
    subprocess.run(["cmake", "-B", "build"])
    subprocess.run(["cmake", "--build", "build", "--parallel", "4"])


def run_test(program: str, *extra_args):
    """run test program.

    Args:
        program: the program to be tested.
        extra_args: extra arguments to the program (for num_cpus).
    """
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    name = program.split("/")[-1]
    print(f"{HEADER}==== Testing {name} ===={ENDC}")
    subprocess.run(test_lena_cmd(program, *extra_args))
    if comapre_img(LENA_OUT, LENA_REF):
        print(f"{OKGREEN}Test {name} passed{ENDC}\n")
    else:
        print(f"{FAIL}Test {name} failed!{ENDC}\n")


def test_lena_cmd(program: str, *extra_args):
    """the command to run program with lena.jpg as test image."""
    cmd = program.split(" ")
    cmd +=[LENA_IN, LENA_OUT]
    cmd += [str(s) for s in extra_args]
    return cmd


def load_img(img: str) -> np.array:
    """read jpg as numpy array."""
    image = Image.open(img)
    return np.asarray(image).astype(int)


def comapre_img(test_img: str, ref_img: str, tol: int = TOL) -> bool:
    """compare two images (range 0-255), with a given tol.

    Args:
        test_img: path to the image to be tested
        ref_img: path to the reference image
        tol: maximum tolerance

    Returns:
        whether two images are the same
    """
    test_img = load_img(test_img)
    ref_img = load_img(ref_img)
    print(f"max diff = {np.max(np.abs(test_img - ref_img))} (tol = {tol})")
    return np.allclose(test_img, ref_img, atol=tol)


if __name__ == "__main__":
    build()
    run_test(sequential_PartB)
    run_test(simd_PartB)
    run_test(pthread_PartB, 4)
    run_test(pthread_simd_PartB, 4)
    run_test(openmp_PartB, 4)
    run_test(cuda_PartB)
    run_test(openacc_PartB)
    run_test(mpi_PartB)
