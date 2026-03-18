## 2024-05-24 - NumPy Array Broadcasting for Pairwise Distances
**Learning:** Using NumPy broadcasting to compute pairwise Euclidean distances between a large array of points (e.g., 262,144 pixels) and a smaller set (e.g., 256 colormap entries) like `diff = pixels[:, np.newaxis, :] - TURBO_COLORMAP[np.newaxis, :, :]` is incredibly memory-inefficient and slow. It creates a massive intermediate array in memory (`H*W*256*3` floats). For finding the minimum distance, evaluating `||C||^2 - 2(P dot C)` using `np.dot` instead of `||P - C||^2` completely avoids allocating this large array and is significantly faster (~40x speedup).
**Action:** When computing pairwise Euclidean distances for searching/nearest-neighbor lookup, use matrix multiplication `P dot C^T` combined with precomputed norms `||C||^2`, skipping `||P||^2` if only the minimum/argmin over `C` is needed. This avoids memory bottlenecks from array broadcasting.

## 2025-03-02 - Array Reshaping and Distance Argmax Optimization
**Learning:** Unrolling channels from a NumPy array and using `np.stack` creates unnecessary intermediate arrays and is slower than leveraging `reshape` and `transpose` directly. Additionally, when searching for the nearest Euclidean neighbor, mathematically transforming the minimization of `||C||^2 - 2(P dot C)` to the maximization of `(P dot C) - 0.5 * ||C||^2` allows the use of `np.argmax`, which can be slightly faster and avoids allocating intermediate distance arrays.
**Action:** Always prefer direct array `reshape` and `transpose` over manual unraveling (`ravel()`) and `stack` for channel manipulation. When calculating argmin distance metrics, evaluate whether multiplying by -0.5 and using `np.argmax` allows avoiding matrix allocations.

## 2025-03-03 - NumPy to Python List Conversion Overhead
**Learning:** Converting a large NumPy array (e.g., 262,144 elements) to a Python `list[float]` and back to a `np.float32` array using `.tolist()` and `np.array()` takes ~2.3 seconds per request. This overhead is incredibly high and unnecessary when the final format needed is bytes. Returning the NumPy array directly from processing functions completely bypasses this performance bottleneck, reducing conversion time to mere milliseconds.
**Action:** Never convert large NumPy arrays to Python lists if the data ultimately stays in a numerical format or gets converted back to an array/bytes. Always maintain the data as a NumPy array and use array operations directly (e.g., `.astype()`, `.tobytes()`).

## 2025-03-04 - gzip.compress default level is incredibly slow
**Learning:** Python's `gzip.compress` defaults to `compresslevel=9` (maximum compression). For reasonably sized binary arrays (e.g., 1MB float32 array converted to bytes), compression level 9 is incredibly slow (~500ms) but only yields a marginally smaller output compared to compression level 1 (~25ms). This results in massive API response latency with almost no bandwidth saving.
**Action:** When compressing API payloads dynamically (especially numpy array bytes), always explicitly specify `compresslevel=1` to optimize for speed over marginal compression size differences.

## 2025-03-05 - Array Type Conversion Overhead
**Learning:** Expanding dimensions on an array (e.g. `np.expand_dims(data, axis=0).astype(np.float32)`) implicitly allocates a new array if `copy=False` isn't specified, even when the original array is already of type `np.float32`. This creates significant memory overhead and allocation time (~0.3ms vs ~0.003ms for a 3x512x512 array).
**Action:** When casting types for inputs that may already be of the target type, always use `copy=False` or check the dtype explicitly before casting.

## 2025-03-05 - 1D Array Lookup Optimization for 3D Colors
**Learning:** Using a 3D uint8 array lookup `COLORMAP_LUT[R, G, B]` followed by multiplication `(indices * (15.0 / 256)).astype(np.float32)` is memory intensive and computationally expensive for large image arrays. Pre-computing a flattened 1D `float32` array and looking up the values using a 32-bit integer array view `(padded.view('<u4'))` drastically improves lookup speed, avoiding both multidimensional indexing and runtime math operations.
**Action:** When mapping discrete RGB combinations to float values in large arrays, pad to RGBA, interpret as little-endian `<u4` integers via `.view()`, and map directly to a pre-computed 1D float array.

## 2025-03-05 - In-Place Array Operations
**Learning:** Normalization and denormalization logic like `(data + 1.0) * 127.5` and `np.clip(...)` creates multiple massive intermediate arrays. This is extremely inefficient and puts heavy pressure on the garbage collector. Doing `np.multiply(raw, 127.5, out=denorm)` combined with `np.add` and `np.clip(..., out=denorm)` prevents allocating these intermediate arrays, bringing a ~20% speedup.
**Action:** Use numpy in-place arithmetic operations (`out=`) for pixel-wise math whenever operating on large image data buffers to prevent allocating large intermediate float arrays.

## 2025-03-05 - Mutating ONNX Outputs In-Place
**Learning:** ONNX runtime python bindings return standard mutable numpy arrays. Creating an `empty_like` array to store results of array arithmetic when mapping outputs back to image spaces is unnecessary and costs ~3MB per request plus allocation time.
**Action:** Always perform in-place mathematical operations directly on the ONNX output array where possible, rather than pre-allocating an `empty_like` array, avoiding extra `float32` array memory allocations.

## 2025-03-05 - FastAPI Async CPU Blocking
**Learning:** Using `async def` for FastAPI endpoints containing CPU-heavy operations (e.g. gzip compression, base64 encoding, numpy math, synchronous ONNX inference) runs the code directly on the single event loop. This blocks the server from processing concurrent requests, severely impacting API concurrency and causing `/health` checks to time out under load.
**Action:** When a FastAPI endpoint consists primarily of CPU-bound, synchronous third-party library calls, define it using standard `def` instead of `async def`. FastAPI will automatically run `def` endpoints in an external threadpool, preserving the event loop's responsiveness.

## 2025-03-05 - Parallelizing NumPy Matmul Startup
**Learning:** `np.matmul` natively releases the Python Global Interpreter Lock (GIL). When computing many independent matrix multiplications iteratively on a single thread (like generating a large 1D lookup table for colormaps), using `concurrent.futures.ThreadPoolExecutor` allows full utilization of multi-core CPUs.
**Action:** When performing heavy independent matrix math in a loop that blocks server startup, divide the work into chunks and use a standard Python ThreadPoolExecutor to speed it up. Be mindful of temporary array memory consumption per thread and cap the number of threads (e.g. `min(8, os.cpu_count())`).

## 2025-03-05 - Avoid .tobytes() zero-copy bytes memory compression
**Learning:** Using `.tobytes()` on a numpy array creates a full copy of the array's data as a byte string. For a 1MB float32 array, this takes up extra memory and time. `gzip.compress` supports buffer protocol objects, so we can pass `memoryview(arr)` directly to compress the array data without copying it into a python `bytes` object first.
**Action:** When feeding numpy arrays to functions accepting buffers (like `gzip.compress`), always wrap them in a `memoryview` instead of calling `.tobytes()` to eliminate memory copies and improve latency.

## 2025-03-05 - NumPy Transpose and Astype Order
**Learning:** Calling `.astype(np.uint8).transpose(...)` creates an intermediate uint8 array in the original layout, then returns a non-contiguous transposed view. By calling `.transpose(...).astype(np.uint8)`, numpy iterates over the transposed memory view and constructs a brand new, memory-contiguous output array directly.
**Action:** When a contiguous output array is needed (like for PIL conversion or memory operations), transpose the array *before* casting its type to force contiguous memory layout generation.

## 2025-03-05 - Avoid np.zeros for Padded Structs
**Learning:** Allocating an array with `np.zeros` and immediately overwriting most of it is inefficient. Using `np.empty` and then explicitly zeroing only the required padding parts (like the alpha channel in an RGBA padding structure) skips a full memory-zeroing pass.
**Action:** Use `np.empty` and explicitly initialize required values instead of `np.zeros` when building a temporary data structure that gets fully overwritten, especially in hot loops or large arrays.

## 2025-03-05 - Factoring out partial Matrix Multiplications from Loops
**Learning:** Computing the dot product `P dot C` inside a loop where a subset of `P` dimensions remains constant across iterations leads to massive redundant computation. Evaluating `G * C_g + B * C_b` inside a thread pool for every `R` value in a 256x256x256 lookup table generation wastes gigabytes of matrix multiplication bandwidth and drastically increases startup time.
**Action:** Always identify components of a dot product or matrix multiplication that are constant with respect to an outer loop. Factor them out and pre-calculate them once before the loop. Inside the inner loop, use a simple `np.add` to combine the pre-calculated term with the loop-dependent term (`R * C_r`), replacing expensive $O(N \times 3 \times C)$ operations with extremely fast $O(N \times C)$ array additions.

## 2025-03-05 - Concurrent Endpoint Processing
**Learning:** In endpoints executing heavy synchronous I/O or GIL-releasing work (like `gzip.compress` or PIL `Image.save`), sequential execution leaves resources idle and unnecessarily increases latency. Even though FastAPI runs standard `def` endpoints in an external thread pool, the endpoint itself is still single-threaded, and executing two 45ms tasks sequentially takes 90ms.
**Action:** When an endpoint performs multiple slow, independent, GIL-releasing operations (e.g. gzip compression and image encoding), wrap them in a local `concurrent.futures.ThreadPoolExecutor` to process them in parallel, halving payload preparation time.
