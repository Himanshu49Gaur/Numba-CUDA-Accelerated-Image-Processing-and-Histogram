# Project Description

This project focuses on translating and optimizing compute-intensive algorithms—specifically, an image blur and a histogram calculation—from CPU to GPU using Python and Numba's CUDA toolkit. The objective is to demonstrate how to write GPU kernels in Python, configure parallel computation, and implement techniques like shared-memory tiling and atomic operations to maximize performance. The project includes performance benchmarking to analyze the speedups gained by using Numba-CUDA.

### Key Features

### Image Blur Kernel:
Implements a 9x9 average image blur using a @cuda.jit decorated Python function. The kernel is designed to handle each pixel with a dedicated thread, and it can also compute a stride for when the grid is smaller than the image dimensions.

### Shared-Memory Tiling Optimization:
Features an enhanced image blur kernel that utilizes fast shared memory to load tiles and their borders before averaging. This technique is used to minimize global memory bottlenecks and maximize throughput.

### Histogram Kernel:
A GPU-accelerated histogram kernel that efficiently counts bin values using a per-block shared-memory array and atomic additions. This design ensures thread safety and accurate accumulation of counts from multiple threads processing the data.

### Performance Benchmarking:
The project includes code to benchmark the execution times of the GPU-accelerated kernels against a known-good CPU implementation. This allows for a direct comparison of performance gains.

### Correctness Validation:
The outputs of the kernels are validated against CPU implementations, with the histogram kernel specifically compared against a pure-Python or NumPy-based histogram.

``Installation & Usage``
Prerequisites
A system with an NVIDIA GPU and a compatible CUDA toolkit installed.

Python 3.x

Jupyter Notebook (for running the `.ipynb` file)

Method : Using ``requirements.txt`` 
Clone the repository: 
```
git clone https://github.com/Himanshu49Gaur/Numba-CUDA-Accelerated-Image-Processing-and-Histogram.git
cd Numba-CUDA-Accelerated-Image-Processing-and-Histogram
```

Install the required packages using `pip`:
```
pip install -r requirements.txt
```

Launch Jupyter Notebook and open the `Project.ipynb` file:
```
jupyter notebook Project.ipynb
```

### Running the Project
Once the Jupyter Notebook is open, execute the cells sequentially to run the Image Blur and Histogram kernels.

The notebook will perform the calculations, benchmark the performance, and print the results and validation checks.

### How This Project Helps and What You Can Analyze
This project is an excellent resource for anyone interested in high-performance computing and GPU programming within the Python ecosystem. It provides a platform to analyze several key concepts:

### GPU Programming Fundamentals:
The code demonstrates how to translate C CUDA kernels into Python using @cuda.jit and how to configure thread and block dimensions for parallel execution on the GPU. You can analyze how these configurations affect kernel performance.

### Memory Optimization:
The shared-memory tiling implementation for the image blur provides a clear case study on minimizing memory bottlenecks by leveraging on-chip shared memory. You can analyze how this technique improves performance compared to the non-optimized version.

### Parallel Algorithm Design:
The histogram kernel demonstrates a common parallel pattern involving per-block accumulation and atomic operations to handle concurrent writes to shared and global memory. You can analyze the importance of thread synchronization for ensuring correctness.

### Benchmarking and Performance Analysis:
By comparing GPU and CPU execution times, you can analyze the specific conditions under which GPU acceleration provides significant speedups. This includes understanding the overhead of data transfer and how it influences the overall performance of the application.

### Demos
* **Image Blur:** [Insert an image showing before and after blur]
* **Performance Chart:** [Insert a chart showing CPU vs. GPU performance]

### Troubleshooting
* **Numba/CUDA not working:** Ensure you are running the project on a system with a compatible NVIDIA GPU and have selected the GPU runtime in Google Colab.
* **Installation Issues:** If you encounter `ModuleNotFoundError`, make sure all dependencies from `requirements.txt` are installed.

### Future Enhancements
* Experiment with different block and grid sizes to find the optimal configuration for maximum performance.
* Implement additional image filters (e.g., Gaussian, edge detection) using Numba-CUDA to further explore its capabilities.
* Integrate a command-line interface for running the kernels without the need for a Jupyter Notebook.

### Problem-Solving and Debugging:
The project provides a basis for debugging and validating the results of parallel kernels. The comparison with CPU implementations allows you to identify and troubleshoot subtle discrepancies or errors that can arise in parallel code.

### Disclaimer
This project is for educational and academic purposes only. The user is solely responsible for the use of this code and any outcomes that may result from its implementation.
