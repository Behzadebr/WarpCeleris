import warp as wp

# Kernel to add a constant to each element of a 2D array
@wp.kernel
def add_constant_kernel(arr: wp.array2d(dtype=wp.vec4), constant: float):
    i, j = wp.tid()  # Get the 2D thread index
    arr[i, j] += wp.vec4(constant, constant, constant, constant)  # Add constant to each component