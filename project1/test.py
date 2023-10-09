import subprocess
from typing import List
from PIL import Image
import numpy as np

# test tolerance 15
TOL = 15

HEADER = '\033[95m'
OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

LENA_IN = "./images/Lena-RGB.jpg"
LENA_REF = "./images/Lena-Smooth.jpg"
LENA_OUT = "./images/Lena-test.jpg"

sequential_PartB = "./build/src/cpu/sequential_PartB"
simd_PartB = "./build/src/cpu/simd_PartB"
pthread_PartB = "./build/src/cpu/pthread_PartB"
pthread_simd_PartB = "./build/src/cpu/pthread_simd_PartB"
openmp_PartB = "./build/src/cpu/openmp_PartB"
cuda_PartB = "./build/src/gpu/cuda_PartB"


def build():
    subprocess.run(["cmake", "-B", "build"])
    subprocess.run(["cmake", "--build", "build", "--parallel", "4"])


def run_test_output(program: str, *extra_args):
    print(f"{HEADER}==== Testing {program} ===={ENDC}")
    subprocess.run(test_lena_cmd(program, *extra_args))
    if comapre_img(LENA_OUT, LENA_REF):
        print(f"{OKGREEN}Test {program} passed{ENDC}\n")
    else:
        print(f"{FAIL}Test {program} failed!{ENDC}\n")


def test_lena_cmd(program: str, *extra_args):
    cmd = [program, LENA_IN, LENA_OUT]
    cmd += [str(s) for s in extra_args]
    return cmd


def load_img(img: str) -> np.array:
    image = Image.open(img)
    return np.asarray(image).astype(int)


def comapre_img(test_img: str, ref_img: str) -> bool:
    test_img = load_img(test_img)
    ref_img = load_img(ref_img)
    print(f"max diff = {np.max(np.abs(test_img - ref_img))} (tol = {TOL})")
    return np.allclose(test_img, ref_img, atol=TOL)


if __name__ == "__main__":
    build()
    run_test_output(sequential_PartB)
    run_test_output(simd_PartB)
    run_test_output(pthread_PartB, 4)
    run_test_output(pthread_simd_PartB, 4)
    run_test_output(openmp_PartB, 4)
    run_test_output(cuda_PartB)
