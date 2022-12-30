module LibMagma

to_c_type(t::Type) = t
to_c_type_pairs(va_list) = map(enumerate(to_c_type.(va_list))) do (ind, type)
    :(va_list[$ind]::$type)
end

const magma_int_t = Cint

"""
MIC and CUDA use regular pointers on GPU
"""
const magma_ptr = Ptr{Cvoid}

"""
opaque queue structure
"""
mutable struct magma_queue end

const magma_queue_t = Ptr{magma_queue}

"""
    magma_setvector_internal(n, elemSize, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
void magma_setvector_internal( magma_int_t n, magma_int_t elemSize, const void *hx_src, magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_setvector_internal(n, elemSize, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_setvector_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

const magma_const_ptr = Ptr{Cvoid}

"""
    magma_getvector_internal(n, elemSize, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
void magma_getvector_internal( magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src, magma_int_t incx, void *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_getvector_internal(n, elemSize, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_getvector_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_copyvector_internal(n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
void magma_copyvector_internal( magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src, magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_copyvector_internal(n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_copyvector_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_setvector_async_internal(n, elemSize, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
void magma_setvector_async_internal( magma_int_t n, magma_int_t elemSize, const void *hx_src, magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_setvector_async_internal(n, elemSize, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_setvector_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_getvector_async_internal(n, elemSize, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
void magma_getvector_async_internal( magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src, magma_int_t incx, void *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_getvector_async_internal(n, elemSize, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_getvector_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_copyvector_async_internal(n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
void magma_copyvector_async_internal( magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src, magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_copyvector_async_internal(n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_copyvector_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_setmatrix_internal(m, n, elemSize, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
void magma_setmatrix_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, const void *hA_src, magma_int_t lda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_setmatrix_internal(m, n, elemSize, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_setmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_getmatrix_internal(m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
void magma_getmatrix_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src, magma_int_t ldda, void *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_getmatrix_internal(m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_getmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_copymatrix_internal(m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
void magma_copymatrix_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src, magma_int_t ldda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_copymatrix_internal(m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_copymatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_setmatrix_async_internal(m, n, elemSize, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
void magma_setmatrix_async_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, const void *hA_src, magma_int_t lda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_setmatrix_async_internal(m, n, elemSize, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_setmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_getmatrix_async_internal(m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
void magma_getmatrix_async_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src, magma_int_t ldda, void *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_getmatrix_async_internal(m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_getmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_copymatrix_async_internal(m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
void magma_copymatrix_async_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src, magma_int_t ldda, magma_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_copymatrix_async_internal(m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_copymatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, magma_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

const magmaInt_ptr = Ptr{magma_int_t}

"""
    magma_isetvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_isetvector_internal( magma_int_t n, const magma_int_t *hx_src, magma_int_t incx, magmaInt_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_isetvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_isetvector_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

const magmaInt_const_ptr = Ptr{magma_int_t}

"""
    magma_igetvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_igetvector_internal( magma_int_t n, magmaInt_const_ptr dx_src, magma_int_t incx, magma_int_t *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_igetvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_igetvector_internal, libmagma), Cvoid, (magma_int_t, magmaInt_const_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_icopyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_icopyvector_internal( magma_int_t n, magmaInt_const_ptr dx_src, magma_int_t incx, magmaInt_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_icopyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_icopyvector_internal, libmagma), Cvoid, (magma_int_t, magmaInt_const_ptr, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_isetvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_isetvector_async_internal( magma_int_t n, const magma_int_t *hx_src, magma_int_t incx, magmaInt_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_isetvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_isetvector_async_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_igetvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_igetvector_async_internal( magma_int_t n, magmaInt_const_ptr dx_src, magma_int_t incx, magma_int_t *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_igetvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_igetvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaInt_const_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_icopyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_icopyvector_async_internal( magma_int_t n, magmaInt_const_ptr dx_src, magma_int_t incx, magmaInt_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_icopyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_icopyvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaInt_const_ptr, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_isetmatrix_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_isetmatrix_internal( magma_int_t m, magma_int_t n, const magma_int_t *hA_src, magma_int_t lda, magmaInt_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_isetmatrix_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_isetmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_igetmatrix_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
static inline void magma_igetmatrix_internal( magma_int_t m, magma_int_t n, magmaInt_const_ptr dA_src, magma_int_t ldda, magma_int_t *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_igetmatrix_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_igetmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_icopymatrix_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_icopymatrix_internal( magma_int_t m, magma_int_t n, magmaInt_const_ptr dA_src, magma_int_t ldda, magmaInt_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_icopymatrix_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_icopymatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_isetmatrix_async_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_isetmatrix_async_internal( magma_int_t m, magma_int_t n, const magma_int_t *hA_src, magma_int_t lda, magmaInt_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_isetmatrix_async_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_isetmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_igetmatrix_async_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
static inline void magma_igetmatrix_async_internal( magma_int_t m, magma_int_t n, magmaInt_const_ptr dA_src, magma_int_t ldda, magma_int_t *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_igetmatrix_async_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_igetmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_icopymatrix_async_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_icopymatrix_async_internal( magma_int_t m, magma_int_t n, magmaInt_const_ptr dA_src, magma_int_t ldda, magmaInt_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_icopymatrix_async_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_icopymatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t, magmaInt_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

const magma_index_t = Cint

const magmaIndex_ptr = Ptr{magma_index_t}

"""
    magma_index_setvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_index_setvector_internal( magma_int_t n, const magma_index_t *hx_src, magma_int_t incx, magmaIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_setvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_index_setvector_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_index_t}, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

const magmaIndex_const_ptr = Ptr{magma_index_t}

"""
    magma_index_getvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_index_getvector_internal( magma_int_t n, magmaIndex_const_ptr dx_src, magma_int_t incx, magma_index_t *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_getvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_index_getvector_internal, libmagma), Cvoid, (magma_int_t, magmaIndex_const_ptr, magma_int_t, Ptr{magma_index_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_index_copyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_index_copyvector_internal( magma_int_t n, magmaIndex_const_ptr dx_src, magma_int_t incx, magmaIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_copyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_index_copyvector_internal, libmagma), Cvoid, (magma_int_t, magmaIndex_const_ptr, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_index_setvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_index_setvector_async_internal( magma_int_t n, const magma_index_t *hx_src, magma_int_t incx, magmaIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_setvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_index_setvector_async_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_index_t}, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_index_getvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_index_getvector_async_internal( magma_int_t n, magmaIndex_const_ptr dx_src, magma_int_t incx, magma_index_t *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_getvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_index_getvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaIndex_const_ptr, magma_int_t, Ptr{magma_index_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_index_copyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_index_copyvector_async_internal( magma_int_t n, magmaIndex_const_ptr dx_src, magma_int_t incx, magmaIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_copyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_index_copyvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaIndex_const_ptr, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

const magma_uindex_t = Cuint

const magmaUIndex_ptr = Ptr{magma_uindex_t}

"""
    magma_uindex_setvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_uindex_setvector_internal( magma_int_t n, const magma_uindex_t *hx_src, magma_int_t incx, magmaUIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_uindex_setvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_uindex_setvector_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_uindex_t}, magma_int_t, magmaUIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

const magmaUIndex_const_ptr = Ptr{magma_uindex_t}

"""
    magma_uindex_getvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_uindex_getvector_internal( magma_int_t n, magmaUIndex_const_ptr dx_src, magma_int_t incx, magma_uindex_t *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_uindex_getvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_uindex_getvector_internal, libmagma), Cvoid, (magma_int_t, magmaUIndex_const_ptr, magma_int_t, Ptr{magma_uindex_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_uindex_copyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_uindex_copyvector_internal( magma_int_t n, magmaUIndex_const_ptr dx_src, magma_int_t incx, magmaUIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_uindex_copyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_uindex_copyvector_internal, libmagma), Cvoid, (magma_int_t, magmaUIndex_const_ptr, magma_int_t, magmaUIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_uindex_setvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_uindex_setvector_async_internal( magma_int_t n, const magma_uindex_t *hx_src, magma_int_t incx, magmaUIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_uindex_setvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_uindex_setvector_async_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_uindex_t}, magma_int_t, magmaUIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_uindex_getvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_uindex_getvector_async_internal( magma_int_t n, magmaUIndex_const_ptr dx_src, magma_int_t incx, magma_uindex_t *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_uindex_getvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_uindex_getvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaUIndex_const_ptr, magma_int_t, Ptr{magma_uindex_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_uindex_copyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_uindex_copyvector_async_internal( magma_int_t n, magmaUIndex_const_ptr dx_src, magma_int_t incx, magmaUIndex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_uindex_copyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_uindex_copyvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaUIndex_const_ptr, magma_int_t, magmaUIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_index_setmatrix_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_index_setmatrix_internal( magma_int_t m, magma_int_t n, const magma_index_t *hA_src, magma_int_t lda, magmaIndex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_setmatrix_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_index_setmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_index_t}, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_index_getmatrix_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
static inline void magma_index_getmatrix_internal( magma_int_t m, magma_int_t n, magmaIndex_const_ptr dA_src, magma_int_t ldda, magma_index_t *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_getmatrix_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_index_getmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaIndex_const_ptr, magma_int_t, Ptr{magma_index_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_index_copymatrix_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_index_copymatrix_internal( magma_int_t m, magma_int_t n, magmaIndex_const_ptr dA_src, magma_int_t ldda, magmaIndex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_copymatrix_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_index_copymatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaIndex_const_ptr, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_index_setmatrix_async_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_index_setmatrix_async_internal( magma_int_t m, magma_int_t n, const magma_index_t *hA_src, magma_int_t lda, magmaIndex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_setmatrix_async_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_index_setmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_index_t}, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_index_getmatrix_async_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
static inline void magma_index_getmatrix_async_internal( magma_int_t m, magma_int_t n, magmaIndex_const_ptr dA_src, magma_int_t ldda, magma_index_t *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_getmatrix_async_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_index_getmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaIndex_const_ptr, magma_int_t, Ptr{magma_index_t}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_index_copymatrix_async_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_index_copymatrix_async_internal( magma_int_t m, magma_int_t n, magmaIndex_const_ptr dA_src, magma_int_t ldda, magmaIndex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_index_copymatrix_async_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_index_copymatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaIndex_const_ptr, magma_int_t, magmaIndex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

const magmaDoubleComplex = Cint

const magmaDoubleComplex_ptr = Ptr{magmaDoubleComplex}

"""
    magma_zsetvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_zsetvector_internal( magma_int_t n, magmaDoubleComplex const *hx_src, magma_int_t incx, magmaDoubleComplex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zsetvector_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_zsetvector_internal, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

const magmaDoubleComplex_const_ptr = Ptr{magmaDoubleComplex}

"""
    magma_zgetvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_zgetvector_internal( magma_int_t n, magmaDoubleComplex_const_ptr dx_src, magma_int_t incx, magmaDoubleComplex *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zgetvector_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_zgetvector_internal, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_zcopyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_zcopyvector_internal( magma_int_t n, magmaDoubleComplex_const_ptr dx_src, magma_int_t incx, magmaDoubleComplex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zcopyvector_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_zcopyvector_internal, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_zsetvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_zsetvector_async_internal( magma_int_t n, magmaDoubleComplex const *hx_src, magma_int_t incx, magmaDoubleComplex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zsetvector_async_internal(n, hx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_zsetvector_async_internal, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_zgetvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_zgetvector_async_internal( magma_int_t n, magmaDoubleComplex_const_ptr dx_src, magma_int_t incx, magmaDoubleComplex *hy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zgetvector_async_internal(n, dx_src, incx, hy_dst, incy, queue, func, file, line)
    ccall((:magma_zgetvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, queue, func, file, line)
end

"""
    magma_zcopyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)


### Prototype
```c
static inline void magma_zcopyvector_async_internal( magma_int_t n, magmaDoubleComplex_const_ptr dx_src, magma_int_t incx, magmaDoubleComplex_ptr dy_dst, magma_int_t incy, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zcopyvector_async_internal(n, dx_src, incx, dy_dst, incy, queue, func, file, line)
    ccall((:magma_zcopyvector_async_internal, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, queue, func, file, line)
end

"""
    magma_zsetmatrix_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_zsetmatrix_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex const *hA_src, magma_int_t lda, magmaDoubleComplex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zsetmatrix_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_zsetmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_zgetmatrix_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
static inline void magma_zgetmatrix_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda, magmaDoubleComplex *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zgetmatrix_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_zgetmatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_zcopymatrix_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_zcopymatrix_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda, magmaDoubleComplex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zcopymatrix_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_zcopymatrix_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_zsetmatrix_async_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_zsetmatrix_async_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex const *hA_src, magma_int_t lda, magmaDoubleComplex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zsetmatrix_async_internal(m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_zsetmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_zgetmatrix_async_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)


### Prototype
```c
static inline void magma_zgetmatrix_async_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda, magmaDoubleComplex *hB_dst, magma_int_t ldb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zgetmatrix_async_internal(m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
    ccall((:magma_zgetmatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, queue, func, file, line)
end

"""
    magma_zcopymatrix_async_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)


### Prototype
```c
static inline void magma_zcopymatrix_async_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda, magmaDoubleComplex_ptr dB_dst, magma_int_t lddb, magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_zcopymatrix_async_internal(m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
    ccall((:magma_zcopymatrix_async_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, queue, func, file, line)
end

"""
    magma_free_internal(ptr, func, file, line)


### Prototype
```c
magma_int_t magma_free_internal( magma_ptr ptr, const char* func, const char* file, int line );
```
"""
function magma_free_internal(ptr, func, file, line)
    ccall((:magma_free_internal, libmagma), magma_int_t, (magma_ptr, Ptr{Cchar}, Ptr{Cchar}, Cint), ptr, func, file, line)
end

"""
    magma_free_pinned_internal(ptr, func, file, line)


### Prototype
```c
magma_int_t magma_free_pinned_internal( void *ptr, const char* func, const char* file, int line );
```
"""
function magma_free_pinned_internal(ptr, func, file, line)
    ccall((:magma_free_pinned_internal, libmagma), magma_int_t, (Ptr{Cvoid}, Ptr{Cchar}, Ptr{Cchar}, Cint), ptr, func, file, line)
end

const magma_device_t = magma_int_t

"""
    magma_queue_create_internal(device, queue_ptr, func, file, line)


### Prototype
```c
void magma_queue_create_internal( magma_device_t device, magma_queue_t* queue_ptr, const char* func, const char* file, int line );
```
"""
function magma_queue_create_internal(device, queue_ptr, func, file, line)
    ccall((:magma_queue_create_internal, libmagma), Cvoid, (magma_device_t, Ptr{magma_queue_t}, Ptr{Cchar}, Ptr{Cchar}, Cint), device, queue_ptr, func, file, line)
end

"""
    magma_queue_create_from_cuda_internal(device, stream, cublas, cusparse, queue_ptr, func, file, line)


### Prototype
```c
void magma_queue_create_from_cuda_internal( magma_device_t device, cudaStream_t stream, cublasHandle_t cublas, cusparseHandle_t cusparse, magma_queue_t* queue_ptr, const char* func, const char* file, int line );
```
"""
function magma_queue_create_from_cuda_internal(device, stream, cublas, cusparse, queue_ptr, func, file, line)
    ccall((:magma_queue_create_from_cuda_internal, libmagma), Cvoid, (magma_device_t, Cint, Cint, Cint, Ptr{magma_queue_t}, Ptr{Cchar}, Ptr{Cchar}, Cint), device, stream, cublas, cusparse, queue_ptr, func, file, line)
end

"""
    magma_queue_destroy_internal(queue, func, file, line)


### Prototype
```c
void magma_queue_destroy_internal( magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_queue_destroy_internal(queue, func, file, line)
    ccall((:magma_queue_destroy_internal, libmagma), Cvoid, (magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), queue, func, file, line)
end

"""
    magma_queue_sync_internal(queue, func, file, line)


### Prototype
```c
void magma_queue_sync_internal( magma_queue_t queue, const char* func, const char* file, int line );
```
"""
function magma_queue_sync_internal(queue, func, file, line)
    ccall((:magma_queue_sync_internal, libmagma), Cvoid, (magma_queue_t, Ptr{Cchar}, Ptr{Cchar}, Cint), queue, func, file, line)
end

"""
    magma_setvector_v1_internal(n, elemSize, hx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
void magma_setvector_v1_internal( magma_int_t n, magma_int_t elemSize, const void *hx_src, magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_setvector_v1_internal(n, elemSize, hx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_setvector_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, hx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_getvector_v1_internal(n, elemSize, dx_src, incx, hy_dst, incy, func, file, line)


### Prototype
```c
void magma_getvector_v1_internal( magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src, magma_int_t incx, void *hy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_getvector_v1_internal(n, elemSize, dx_src, incx, hy_dst, incy, func, file, line)
    ccall((:magma_getvector_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, Ptr{Cvoid}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, dx_src, incx, hy_dst, incy, func, file, line)
end

"""
    magma_copyvector_v1_internal(n, elemSize, dx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
void magma_copyvector_v1_internal( magma_int_t n, magma_int_t elemSize, magma_const_ptr dx_src, magma_int_t incx, magma_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_copyvector_v1_internal(n, elemSize, dx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_copyvector_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, magma_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, elemSize, dx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_setmatrix_v1_internal(m, n, elemSize, hA_src, lda, dB_dst, lddb, func, file, line)


### Prototype
```c
void magma_setmatrix_v1_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, const void *hA_src, magma_int_t lda, magma_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_setmatrix_v1_internal(m, n, elemSize, hA_src, lda, dB_dst, lddb, func, file, line)
    ccall((:magma_setmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, Ptr{Cvoid}, magma_int_t, magma_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, hA_src, lda, dB_dst, lddb, func, file, line)
end

"""
    magma_getmatrix_v1_internal(m, n, elemSize, dA_src, ldda, hB_dst, ldb, func, file, line)


### Prototype
```c
void magma_getmatrix_v1_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src, magma_int_t ldda, void *hB_dst, magma_int_t ldb, const char* func, const char* file, int line );
```
"""
function magma_getmatrix_v1_internal(m, n, elemSize, dA_src, ldda, hB_dst, ldb, func, file, line)
    ccall((:magma_getmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, Ptr{Cvoid}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, dA_src, ldda, hB_dst, ldb, func, file, line)
end

"""
    magma_copymatrix_v1_internal(m, n, elemSize, dA_src, ldda, dB_dst, lddb, func, file, line)


### Prototype
```c
void magma_copymatrix_v1_internal( magma_int_t m, magma_int_t n, magma_int_t elemSize, magma_const_ptr dA_src, magma_int_t ldda, magma_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_copymatrix_v1_internal(m, n, elemSize, dA_src, ldda, dB_dst, lddb, func, file, line)
    ccall((:magma_copymatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_const_ptr, magma_int_t, magma_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, elemSize, dA_src, ldda, dB_dst, lddb, func, file, line)
end

"""
    magma_isetvector_v1_internal(n, hx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_isetvector_v1_internal( magma_int_t n, const magma_int_t *hx_src, magma_int_t incx, magmaInt_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_isetvector_v1_internal(n, hx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_isetvector_v1_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaInt_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_igetvector_v1_internal(n, dx_src, incx, hy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_igetvector_v1_internal( magma_int_t n, magmaInt_const_ptr dx_src, magma_int_t incx, magma_int_t *hy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_igetvector_v1_internal(n, dx_src, incx, hy_dst, incy, func, file, line)
    ccall((:magma_igetvector_v1_internal, libmagma), Cvoid, (magma_int_t, magmaInt_const_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, func, file, line)
end

"""
    magma_icopyvector_v1_internal(n, dx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_icopyvector_v1_internal( magma_int_t n, magmaInt_const_ptr dx_src, magma_int_t incx, magmaInt_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_icopyvector_v1_internal(n, dx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_icopyvector_v1_internal, libmagma), Cvoid, (magma_int_t, magmaInt_const_ptr, magma_int_t, magmaInt_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_isetmatrix_v1_internal(m, n, hA_src, lda, dB_dst, lddb, func, file, line)


### Prototype
```c
static inline void magma_isetmatrix_v1_internal( magma_int_t m, magma_int_t n, const magma_int_t *hA_src, magma_int_t lda, magmaInt_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_isetmatrix_v1_internal(m, n, hA_src, lda, dB_dst, lddb, func, file, line)
    ccall((:magma_isetmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaInt_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, func, file, line)
end

"""
    magma_igetmatrix_v1_internal(m, n, dA_src, ldda, hB_dst, ldb, func, file, line)


### Prototype
```c
static inline void magma_igetmatrix_v1_internal( magma_int_t m, magma_int_t n, magmaInt_const_ptr dA_src, magma_int_t ldda, magma_int_t *hB_dst, magma_int_t ldb, const char* func, const char* file, int line );
```
"""
function magma_igetmatrix_v1_internal(m, n, dA_src, ldda, hB_dst, ldb, func, file, line)
    ccall((:magma_igetmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, func, file, line)
end

"""
    magma_icopymatrix_v1_internal(m, n, dA_src, ldda, dB_dst, lddb, func, file, line)


### Prototype
```c
static inline void magma_icopymatrix_v1_internal( magma_int_t m, magma_int_t n, magmaInt_const_ptr dA_src, magma_int_t ldda, magmaInt_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_icopymatrix_v1_internal(m, n, dA_src, ldda, dB_dst, lddb, func, file, line)
    ccall((:magma_icopymatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t, magmaInt_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, func, file, line)
end

"""
    magma_index_setvector_v1_internal(n, hx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_index_setvector_v1_internal( magma_int_t n, const magma_index_t *hx_src, magma_int_t incx, magmaIndex_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_index_setvector_v1_internal(n, hx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_index_setvector_v1_internal, libmagma), Cvoid, (magma_int_t, Ptr{magma_index_t}, magma_int_t, magmaIndex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_index_getvector_v1_internal(n, dx_src, incx, hy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_index_getvector_v1_internal( magma_int_t n, magmaIndex_const_ptr dx_src, magma_int_t incx, magma_index_t *hy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_index_getvector_v1_internal(n, dx_src, incx, hy_dst, incy, func, file, line)
    ccall((:magma_index_getvector_v1_internal, libmagma), Cvoid, (magma_int_t, magmaIndex_const_ptr, magma_int_t, Ptr{magma_index_t}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, func, file, line)
end

"""
    magma_index_copyvector_v1_internal(n, dx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_index_copyvector_v1_internal( magma_int_t n, magmaIndex_const_ptr dx_src, magma_int_t incx, magmaIndex_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_index_copyvector_v1_internal(n, dx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_index_copyvector_v1_internal, libmagma), Cvoid, (magma_int_t, magmaIndex_const_ptr, magma_int_t, magmaIndex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_index_setmatrix_v1_internal(m, n, hA_src, lda, dB_dst, lddb, func, file, line)


### Prototype
```c
static inline void magma_index_setmatrix_v1_internal( magma_int_t m, magma_int_t n, const magma_index_t *hA_src, magma_int_t lda, magmaIndex_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_index_setmatrix_v1_internal(m, n, hA_src, lda, dB_dst, lddb, func, file, line)
    ccall((:magma_index_setmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_index_t}, magma_int_t, magmaIndex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, func, file, line)
end

"""
    magma_index_getmatrix_v1_internal(m, n, dA_src, ldda, hB_dst, ldb, func, file, line)


### Prototype
```c
static inline void magma_index_getmatrix_v1_internal( magma_int_t m, magma_int_t n, magmaIndex_const_ptr dA_src, magma_int_t ldda, magma_index_t *hB_dst, magma_int_t ldb, const char* func, const char* file, int line );
```
"""
function magma_index_getmatrix_v1_internal(m, n, dA_src, ldda, hB_dst, ldb, func, file, line)
    ccall((:magma_index_getmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaIndex_const_ptr, magma_int_t, Ptr{magma_index_t}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, func, file, line)
end

"""
    magma_index_copymatrix_v1_internal(m, n, dA_src, ldda, dB_dst, lddb, func, file, line)


### Prototype
```c
static inline void magma_index_copymatrix_v1_internal( magma_int_t m, magma_int_t n, magmaIndex_const_ptr dA_src, magma_int_t ldda, magmaIndex_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_index_copymatrix_v1_internal(m, n, dA_src, ldda, dB_dst, lddb, func, file, line)
    ccall((:magma_index_copymatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaIndex_const_ptr, magma_int_t, magmaIndex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, func, file, line)
end

"""
    magma_zsetvector_v1_internal(n, hx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_zsetvector_v1_internal( magma_int_t n, magmaDoubleComplex const *hx_src, magma_int_t incx, magmaDoubleComplex_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_zsetvector_v1_internal(n, hx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_zsetvector_v1_internal, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, hx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_zgetvector_v1_internal(n, dx_src, incx, hy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_zgetvector_v1_internal( magma_int_t n, magmaDoubleComplex_const_ptr dx_src, magma_int_t incx, magmaDoubleComplex *hy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_zgetvector_v1_internal(n, dx_src, incx, hy_dst, incy, func, file, line)
    ccall((:magma_zgetvector_v1_internal, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, hy_dst, incy, func, file, line)
end

"""
    magma_zcopyvector_v1_internal(n, dx_src, incx, dy_dst, incy, func, file, line)


### Prototype
```c
static inline void magma_zcopyvector_v1_internal( magma_int_t n, magmaDoubleComplex_const_ptr dx_src, magma_int_t incx, magmaDoubleComplex_ptr dy_dst, magma_int_t incy, const char* func, const char* file, int line );
```
"""
function magma_zcopyvector_v1_internal(n, dx_src, incx, dy_dst, incy, func, file, line)
    ccall((:magma_zcopyvector_v1_internal, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), n, dx_src, incx, dy_dst, incy, func, file, line)
end

"""
    magma_zsetmatrix_v1_internal(m, n, hA_src, lda, dB_dst, lddb, func, file, line)


### Prototype
```c
static inline void magma_zsetmatrix_v1_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex const *hA_src, magma_int_t lda, magmaDoubleComplex_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_zsetmatrix_v1_internal(m, n, hA_src, lda, dB_dst, lddb, func, file, line)
    ccall((:magma_zsetmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, hA_src, lda, dB_dst, lddb, func, file, line)
end

"""
    magma_zgetmatrix_v1_internal(m, n, dA_src, ldda, hB_dst, ldb, func, file, line)


### Prototype
```c
static inline void magma_zgetmatrix_v1_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda, magmaDoubleComplex *hB_dst, magma_int_t ldb, const char* func, const char* file, int line );
```
"""
function magma_zgetmatrix_v1_internal(m, n, dA_src, ldda, hB_dst, ldb, func, file, line)
    ccall((:magma_zgetmatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, hB_dst, ldb, func, file, line)
end

"""
    magma_zcopymatrix_v1_internal(m, n, dA_src, ldda, dB_dst, lddb, func, file, line)


### Prototype
```c
static inline void magma_zcopymatrix_v1_internal( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda, magmaDoubleComplex_ptr dB_dst, magma_int_t lddb, const char* func, const char* file, int line );
```
"""
function magma_zcopymatrix_v1_internal(m, n, dA_src, ldda, dB_dst, lddb, func, file, line)
    ccall((:magma_zcopymatrix_v1_internal, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cchar}, Ptr{Cchar}, Cint), m, n, dA_src, ldda, dB_dst, lddb, func, file, line)
end

"""
    magma_queue_create_v1_internal(queue_ptr, func, file, line)


### Prototype
```c
void magma_queue_create_v1_internal( magma_queue_t* queue_ptr, const char* func, const char* file, int line );
```
"""
function magma_queue_create_v1_internal(queue_ptr, func, file, line)
    ccall((:magma_queue_create_v1_internal, libmagma), Cvoid, (Ptr{magma_queue_t}, Ptr{Cchar}, Ptr{Cchar}, Cint), queue_ptr, func, file, line)
end

"""
    magmablas_ztranspose_inplace_v1(n, dA, ldda)

Transpose functions
### Prototype
```c
void magmablas_ztranspose_inplace_v1( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magmablas_ztranspose_inplace_v1(n, dA, ldda)
    ccall((:magmablas_ztranspose_inplace_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, dA, ldda)
end

"""
    magmablas_ztranspose_conj_inplace_v1(n, dA, ldda)


### Prototype
```c
void magmablas_ztranspose_conj_inplace_v1( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magmablas_ztranspose_conj_inplace_v1(n, dA, ldda)
    ccall((:magmablas_ztranspose_conj_inplace_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, dA, ldda)
end

"""
    magmablas_ztranspose_v1(m, n, dA, ldda, dAT, lddat)


### Prototype
```c
void magmablas_ztranspose_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dAT, magma_int_t lddat );
```
"""
function magmablas_ztranspose_v1(m, n, dA, ldda, dAT, lddat)
    ccall((:magmablas_ztranspose_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), m, n, dA, ldda, dAT, lddat)
end

"""
    magmablas_ztranspose_conj_v1(m, n, dA, ldda, dAT, lddat)


### Prototype
```c
void magmablas_ztranspose_conj_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dAT, magma_int_t lddat );
```
"""
function magmablas_ztranspose_conj_v1(m, n, dA, ldda, dAT, lddat)
    ccall((:magmablas_ztranspose_conj_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), m, n, dA, ldda, dAT, lddat)
end

"""
    magmablas_zgetmatrix_transpose_v1(m, n, dAT, ldda, hA, lda, dwork, lddwork, nb)


### Prototype
```c
void magmablas_zgetmatrix_transpose_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dAT, magma_int_t ldda, magmaDoubleComplex *hA, magma_int_t lda, magmaDoubleComplex_ptr dwork, magma_int_t lddwork, magma_int_t nb );
```
"""
function magmablas_zgetmatrix_transpose_v1(m, n, dAT, ldda, hA, lda, dwork, lddwork, nb)
    ccall((:magmablas_zgetmatrix_transpose_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t), m, n, dAT, ldda, hA, lda, dwork, lddwork, nb)
end

"""
    magmablas_zsetmatrix_transpose_v1(m, n, hA, lda, dAT, ldda, dwork, lddwork, nb)


### Prototype
```c
void magmablas_zsetmatrix_transpose_v1( magma_int_t m, magma_int_t n, const magmaDoubleComplex *hA, magma_int_t lda, magmaDoubleComplex_ptr dAT, magma_int_t ldda, magmaDoubleComplex_ptr dwork, magma_int_t lddwork, magma_int_t nb );
```
"""
function magmablas_zsetmatrix_transpose_v1(m, n, hA, lda, dAT, ldda, dwork, lddwork, nb)
    ccall((:magmablas_zsetmatrix_transpose_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t), m, n, hA, lda, dAT, ldda, dwork, lddwork, nb)
end

"""
    magmablas_zprbt_v1(n, dA, ldda, du, dv)

RBT-related functions
### Prototype
```c
void magmablas_zprbt_v1( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr du, magmaDoubleComplex_ptr dv );
```
"""
function magmablas_zprbt_v1(n, dA, ldda, du, dv)
    ccall((:magmablas_zprbt_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr), n, dA, ldda, du, dv)
end

"""
    magmablas_zprbt_mv_v1(n, dv, db)


### Prototype
```c
void magmablas_zprbt_mv_v1( magma_int_t n, magmaDoubleComplex_ptr dv, magmaDoubleComplex_ptr db );
```
"""
function magmablas_zprbt_mv_v1(n, dv, db)
    ccall((:magmablas_zprbt_mv_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr), n, dv, db)
end

"""
    magmablas_zprbt_mtv_v1(n, du, db)


### Prototype
```c
void magmablas_zprbt_mtv_v1( magma_int_t n, magmaDoubleComplex_ptr du, magmaDoubleComplex_ptr db );
```
"""
function magmablas_zprbt_mtv_v1(n, du, db)
    ccall((:magmablas_zprbt_mtv_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr), n, du, db)
end

"""
    magma_zgetmatrix_1D_col_bcyclic_v1(m, n, dA, ldda, hA, lda, ngpu, nb)

Multi-GPU copy functions
### Prototype
```c
void magma_zgetmatrix_1D_col_bcyclic_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr const dA[], magma_int_t ldda, magmaDoubleComplex *hA, magma_int_t lda, magma_int_t ngpu, magma_int_t nb );
```
"""
function magma_zgetmatrix_1D_col_bcyclic_v1(m, n, dA, ldda, hA, lda, ngpu, nb)
    ccall((:magma_zgetmatrix_1D_col_bcyclic_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t), m, n, dA, ldda, hA, lda, ngpu, nb)
end

"""
    magma_zsetmatrix_1D_col_bcyclic_v1(m, n, hA, lda, dA, ldda, ngpu, nb)


### Prototype
```c
void magma_zsetmatrix_1D_col_bcyclic_v1( magma_int_t m, magma_int_t n, const magmaDoubleComplex *hA, magma_int_t lda, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t ngpu, magma_int_t nb );
```
"""
function magma_zsetmatrix_1D_col_bcyclic_v1(m, n, hA, lda, dA, ldda, ngpu, nb)
    ccall((:magma_zsetmatrix_1D_col_bcyclic_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t), m, n, hA, lda, dA, ldda, ngpu, nb)
end

"""
    magma_zgetmatrix_1D_row_bcyclic_v1(m, n, dA, ldda, hA, lda, ngpu, nb)


### Prototype
```c
void magma_zgetmatrix_1D_row_bcyclic_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr const dA[], magma_int_t ldda, magmaDoubleComplex *hA, magma_int_t lda, magma_int_t ngpu, magma_int_t nb );
```
"""
function magma_zgetmatrix_1D_row_bcyclic_v1(m, n, dA, ldda, hA, lda, ngpu, nb)
    ccall((:magma_zgetmatrix_1D_row_bcyclic_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t), m, n, dA, ldda, hA, lda, ngpu, nb)
end

"""
    magma_zsetmatrix_1D_row_bcyclic_v1(m, n, hA, lda, dA, ldda, ngpu, nb)


### Prototype
```c
void magma_zsetmatrix_1D_row_bcyclic_v1( magma_int_t m, magma_int_t n, const magmaDoubleComplex *hA, magma_int_t lda, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t ngpu, magma_int_t nb );
```
"""
function magma_zsetmatrix_1D_row_bcyclic_v1(m, n, hA, lda, dA, ldda, ngpu, nb)
    ccall((:magma_zsetmatrix_1D_row_bcyclic_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t), m, n, hA, lda, dA, ldda, ngpu, nb)
end

"""
    magmablas_zgeadd_v1(m, n, alpha, dA, ldda, dB, lddb)

LAPACK auxiliary functions (alphabetical order)
### Prototype
```c
void magmablas_zgeadd_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magmablas_zgeadd_v1(m, n, alpha, dA, ldda, dB, lddb)
    ccall((:magmablas_zgeadd_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), m, n, alpha, dA, ldda, dB, lddb)
end

"""
    magmablas_zgeadd2_v1(m, n, alpha, dA, ldda, beta, dB, lddb)


### Prototype
```c
void magmablas_zgeadd2_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex beta, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magmablas_zgeadd2_v1(m, n, alpha, dA, ldda, beta, dB, lddb)
    ccall((:magmablas_zgeadd2_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), m, n, alpha, dA, ldda, beta, dB, lddb)
end

@enum magma_uplo_t::UInt32 begin
    MagmaUpper = 121
    MagmaLower = 122
    MagmaFull = 123
    MagmaHessenberg = 124
end

"""
    magmablas_zlacpy_v1(uplo, m, n, dA, ldda, dB, lddb)


### Prototype
```c
void magmablas_zlacpy_v1( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magmablas_zlacpy_v1(uplo, m, n, dA, ldda, dB, lddb)
    ccall((:magmablas_zlacpy_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, m, n, dA, ldda, dB, lddb)
end

"""
    magmablas_zlacpy_conj_v1(n, dA1, lda1, dA2, lda2)


### Prototype
```c
void magmablas_zlacpy_conj_v1( magma_int_t n, magmaDoubleComplex_ptr dA1, magma_int_t lda1, magmaDoubleComplex_ptr dA2, magma_int_t lda2 );
```
"""
function magmablas_zlacpy_conj_v1(n, dA1, lda1, dA2, lda2)
    ccall((:magmablas_zlacpy_conj_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, dA1, lda1, dA2, lda2)
end

"""
    magmablas_zlacpy_sym_in_v1(uplo, m, n, rows, perm, dA, ldda, dB, lddb)


### Prototype
```c
void magmablas_zlacpy_sym_in_v1( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t *rows, magma_int_t *perm, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magmablas_zlacpy_sym_in_v1(uplo, m, n, rows, perm, dA, ldda, dB, lddb)
    ccall((:magmablas_zlacpy_sym_in_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, m, n, rows, perm, dA, ldda, dB, lddb)
end

"""
    magmablas_zlacpy_sym_out_v1(uplo, m, n, rows, perm, dA, ldda, dB, lddb)


### Prototype
```c
void magmablas_zlacpy_sym_out_v1( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t *rows, magma_int_t *perm, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magmablas_zlacpy_sym_out_v1(uplo, m, n, rows, perm, dA, ldda, dB, lddb)
    ccall((:magmablas_zlacpy_sym_out_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, m, n, rows, perm, dA, ldda, dB, lddb)
end

@enum magma_norm_t::UInt32 begin
    MagmaOneNorm = 171
    MagmaRealOneNorm = 172
    MagmaTwoNorm = 173
    MagmaFrobeniusNorm = 174
    MagmaInfNorm = 175
    MagmaRealInfNorm = 176
    MagmaMaxNorm = 177
    MagmaRealMaxNorm = 178
end

const magmaDouble_ptr = Ptr{Cdouble}

"""
    magmablas_zlange_v1(norm, m, n, dA, ldda, dwork, lwork)


### Prototype
```c
double magmablas_zlange_v1( magma_norm_t norm, magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDouble_ptr dwork, magma_int_t lwork );
```
"""
function magmablas_zlange_v1(norm, m, n, dA, ldda, dwork, lwork)
    ccall((:magmablas_zlange_v1, libmagma), Cdouble, (magma_norm_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDouble_ptr, magma_int_t), norm, m, n, dA, ldda, dwork, lwork)
end

"""
    magmablas_zlanhe_v1(norm, uplo, n, dA, ldda, dwork, lwork)


### Prototype
```c
double magmablas_zlanhe_v1( magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDouble_ptr dwork, magma_int_t lwork );
```
"""
function magmablas_zlanhe_v1(norm, uplo, n, dA, ldda, dwork, lwork)
    ccall((:magmablas_zlanhe_v1, libmagma), Cdouble, (magma_norm_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDouble_ptr, magma_int_t), norm, uplo, n, dA, ldda, dwork, lwork)
end

"""
    magmablas_zlansy_v1(norm, uplo, n, dA, ldda, dwork, lwork)


### Prototype
```c
double magmablas_zlansy_v1( magma_norm_t norm, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDouble_ptr dwork, magma_int_t lwork );
```
"""
function magmablas_zlansy_v1(norm, uplo, n, dA, ldda, dwork, lwork)
    ccall((:magmablas_zlansy_v1, libmagma), Cdouble, (magma_norm_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDouble_ptr, magma_int_t), norm, uplo, n, dA, ldda, dwork, lwork)
end

"""
    magmablas_zlarfg_v1(n, dalpha, dx, incx, dtau)


### Prototype
```c
void magmablas_zlarfg_v1( magma_int_t n, magmaDoubleComplex_ptr dalpha, magmaDoubleComplex_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dtau );
```
"""
function magmablas_zlarfg_v1(n, dalpha, dx, incx, dtau)
    ccall((:magmablas_zlarfg_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr), n, dalpha, dx, incx, dtau)
end

const magma_type_t = magma_uplo_t

"""
    magmablas_zlascl_v1(type, kl, ku, cfrom, cto, m, n, dA, ldda, info)


### Prototype
```c
void magmablas_zlascl_v1( magma_type_t type, magma_int_t kl, magma_int_t ku, double cfrom, double cto, magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magmablas_zlascl_v1(type, kl, ku, cfrom, cto, m, n, dA, ldda, info)
    ccall((:magmablas_zlascl_v1, libmagma), Cvoid, (magma_type_t, magma_int_t, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), type, kl, ku, cfrom, cto, m, n, dA, ldda, info)
end

"""
    magmablas_zlascl_2x2_v1(type, m, dW, lddw, dA, ldda, info)


### Prototype
```c
void magmablas_zlascl_2x2_v1( magma_type_t type, magma_int_t m, magmaDoubleComplex_const_ptr dW, magma_int_t lddw, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magmablas_zlascl_2x2_v1(type, m, dW, lddw, dA, ldda, info)
    ccall((:magmablas_zlascl_2x2_v1, libmagma), Cvoid, (magma_type_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), type, m, dW, lddw, dA, ldda, info)
end

const magmaDouble_const_ptr = Ptr{Cdouble}

"""
    magmablas_zlascl2_v1(type, m, n, dD, dA, ldda, info)


### Prototype
```c
void magmablas_zlascl2_v1( magma_type_t type, magma_int_t m, magma_int_t n, magmaDouble_const_ptr dD, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magmablas_zlascl2_v1(type, m, n, dD, dA, ldda, info)
    ccall((:magmablas_zlascl2_v1, libmagma), Cvoid, (magma_type_t, magma_int_t, magma_int_t, magmaDouble_const_ptr, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), type, m, n, dD, dA, ldda, info)
end

"""
    magmablas_zlascl_diag_v1(type, m, n, dD, lddd, dA, ldda, info)


### Prototype
```c
void magmablas_zlascl_diag_v1( magma_type_t type, magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dD, magma_int_t lddd, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magmablas_zlascl_diag_v1(type, m, n, dD, lddd, dA, ldda, info)
    ccall((:magmablas_zlascl_diag_v1, libmagma), Cvoid, (magma_type_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), type, m, n, dD, lddd, dA, ldda, info)
end

"""
    magmablas_zlaset_v1(uplo, m, n, offdiag, diag, dA, ldda)


### Prototype
```c
void magmablas_zlaset_v1( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex offdiag, magmaDoubleComplex diag, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magmablas_zlaset_v1(uplo, m, n, offdiag, diag, dA, ldda)
    ccall((:magmablas_zlaset_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, m, n, offdiag, diag, dA, ldda)
end

"""
    magmablas_zlaset_band_v1(uplo, m, n, k, offdiag, diag, dA, ldda)


### Prototype
```c
void magmablas_zlaset_band_v1( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex offdiag, magmaDoubleComplex diag, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magmablas_zlaset_band_v1(uplo, m, n, k, offdiag, diag, dA, ldda)
    ccall((:magmablas_zlaset_band_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, m, n, k, offdiag, diag, dA, ldda)
end

"""
    magmablas_zlaswp_v1(n, dAT, ldda, k1, k2, ipiv, inci)


### Prototype
```c
void magmablas_zlaswp_v1( magma_int_t n, magmaDoubleComplex_ptr dAT, magma_int_t ldda, magma_int_t k1, magma_int_t k2, const magma_int_t *ipiv, magma_int_t inci );
```
"""
function magmablas_zlaswp_v1(n, dAT, ldda, k1, k2, ipiv, inci)
    ccall((:magmablas_zlaswp_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t), n, dAT, ldda, k1, k2, ipiv, inci)
end

"""
    magmablas_zlaswp2_v1(n, dAT, ldda, k1, k2, d_ipiv, inci)


### Prototype
```c
void magmablas_zlaswp2_v1( magma_int_t n, magmaDoubleComplex_ptr dAT, magma_int_t ldda, magma_int_t k1, magma_int_t k2, magmaInt_const_ptr d_ipiv, magma_int_t inci );
```
"""
function magmablas_zlaswp2_v1(n, dAT, ldda, k1, k2, d_ipiv, inci)
    ccall((:magmablas_zlaswp2_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, magmaInt_const_ptr, magma_int_t), n, dAT, ldda, k1, k2, d_ipiv, inci)
end

"""
    magmablas_zlaswp_sym_v1(n, dA, ldda, k1, k2, ipiv, inci)


### Prototype
```c
void magmablas_zlaswp_sym_v1( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t k1, magma_int_t k2, const magma_int_t *ipiv, magma_int_t inci );
```
"""
function magmablas_zlaswp_sym_v1(n, dA, ldda, k1, k2, ipiv, inci)
    ccall((:magmablas_zlaswp_sym_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t), n, dA, ldda, k1, k2, ipiv, inci)
end

"""
    magmablas_zlaswpx_v1(n, dA, ldx, ldy, k1, k2, ipiv, inci)


### Prototype
```c
void magmablas_zlaswpx_v1( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldx, magma_int_t ldy, magma_int_t k1, magma_int_t k2, const magma_int_t *ipiv, magma_int_t inci );
```
"""
function magmablas_zlaswpx_v1(n, dA, ldx, ldy, k1, k2, ipiv, inci)
    ccall((:magmablas_zlaswpx_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t), n, dA, ldx, ldy, k1, k2, ipiv, inci)
end

"""
    magmablas_zsymmetrize_v1(uplo, m, dA, ldda)


### Prototype
```c
void magmablas_zsymmetrize_v1( magma_uplo_t uplo, magma_int_t m, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magmablas_zsymmetrize_v1(uplo, m, dA, ldda)
    ccall((:magmablas_zsymmetrize_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, m, dA, ldda)
end

"""
    magmablas_zsymmetrize_tiles_v1(uplo, m, dA, ldda, ntile, mstride, nstride)


### Prototype
```c
void magmablas_zsymmetrize_tiles_v1( magma_uplo_t uplo, magma_int_t m, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );
```
"""
function magmablas_zsymmetrize_tiles_v1(uplo, m, dA, ldda, ntile, mstride, nstride)
    ccall((:magmablas_zsymmetrize_tiles_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, magma_int_t), uplo, m, dA, ldda, ntile, mstride, nstride)
end

@enum magma_diag_t::UInt32 begin
    MagmaNonUnit = 131
    MagmaUnit = 132
end

"""
    magmablas_ztrtri_diag_v1(uplo, diag, n, dA, ldda, d_dinvA)


### Prototype
```c
void magmablas_ztrtri_diag_v1( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr d_dinvA );
```
"""
function magmablas_ztrtri_diag_v1(uplo, diag, n, dA, ldda, d_dinvA)
    ccall((:magmablas_ztrtri_diag_v1, libmagma), Cvoid, (magma_uplo_t, magma_diag_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr), uplo, diag, n, dA, ldda, d_dinvA)
end

"""
    magmablas_dznrm2_adjust_v1(k, dxnorm, dc)

to cleanup (alphabetical order)
### Prototype
```c
void magmablas_dznrm2_adjust_v1( magma_int_t k, magmaDouble_ptr dxnorm, magmaDoubleComplex_ptr dc );
```
"""
function magmablas_dznrm2_adjust_v1(k, dxnorm, dc)
    ccall((:magmablas_dznrm2_adjust_v1, libmagma), Cvoid, (magma_int_t, magmaDouble_ptr, magmaDoubleComplex_ptr), k, dxnorm, dc)
end

"""
    magmablas_dznrm2_check_v1(m, n, dA, ldda, dxnorm, dlsticc)


### Prototype
```c
void magmablas_dznrm2_check_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDouble_ptr dxnorm, magmaDouble_ptr dlsticc );
```
"""
function magmablas_dznrm2_check_v1(m, n, dA, ldda, dxnorm, dlsticc)
    ccall((:magmablas_dznrm2_check_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDouble_ptr, magmaDouble_ptr), m, n, dA, ldda, dxnorm, dlsticc)
end

"""
    magmablas_dznrm2_cols_v1(m, n, dA, ldda, dxnorm)


### Prototype
```c
void magmablas_dznrm2_cols_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDouble_ptr dxnorm );
```
"""
function magmablas_dznrm2_cols_v1(m, n, dA, ldda, dxnorm)
    ccall((:magmablas_dznrm2_cols_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDouble_ptr), m, n, dA, ldda, dxnorm)
end

"""
    magmablas_dznrm2_row_check_adjust_v1(k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc)


### Prototype
```c
void magmablas_dznrm2_row_check_adjust_v1( magma_int_t k, double tol, magmaDouble_ptr dxnorm, magmaDouble_ptr dxnorm2, magmaDoubleComplex_ptr dC, magma_int_t lddc, magmaDouble_ptr dlsticc );
```
"""
function magmablas_dznrm2_row_check_adjust_v1(k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc)
    ccall((:magmablas_dznrm2_row_check_adjust_v1, libmagma), Cvoid, (magma_int_t, Cdouble, magmaDouble_ptr, magmaDouble_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDouble_ptr), k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc)
end

@enum magma_side_t::UInt32 begin
    MagmaLeft = 141
    MagmaRight = 142
    MagmaBothSides = 143
end

"""
    magma_trans_t

Magma_ConjTrans is an alias for those rare occasions (zlarfb, zun*, zher*k)
where we want Magma_ConjTrans to convert to MagmaTrans in precision generation.
"""
@enum magma_trans_t::UInt32 begin
    MagmaNoTrans = 111
    MagmaTrans = 112
    MagmaConjTrans = 113
    # Magma_ConjTrans = 113
end

@enum magma_direct_t::UInt32 begin
    MagmaForward = 391
    MagmaBackward = 392
end

@enum magma_storev_t::UInt32 begin
    MagmaColumnwise = 401
    MagmaRowwise = 402
end

"""
    magma_zlarfb_gpu_v1(side, trans, direct, storev, m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork)


### Prototype
```c
magma_int_t magma_zlarfb_gpu_v1( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_const_ptr dV, magma_int_t lddv, magmaDoubleComplex_const_ptr dT, magma_int_t lddt, magmaDoubleComplex_ptr dC, magma_int_t lddc, magmaDoubleComplex_ptr dwork, magma_int_t ldwork );
```
"""
function magma_zlarfb_gpu_v1(side, trans, direct, storev, m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork)
    ccall((:magma_zlarfb_gpu_v1, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_direct_t, magma_storev_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, trans, direct, storev, m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork)
end

"""
    magma_zlarfb_gpu_gemm_v1(side, trans, direct, storev, m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork, dworkvt, ldworkvt)


### Prototype
```c
magma_int_t magma_zlarfb_gpu_gemm_v1( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_const_ptr dV, magma_int_t lddv, magmaDoubleComplex_const_ptr dT, magma_int_t lddt, magmaDoubleComplex_ptr dC, magma_int_t lddc, magmaDoubleComplex_ptr dwork, magma_int_t ldwork, magmaDoubleComplex_ptr dworkvt, magma_int_t ldworkvt );
```
"""
function magma_zlarfb_gpu_gemm_v1(side, trans, direct, storev, m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork, dworkvt, ldworkvt)
    ccall((:magma_zlarfb_gpu_gemm_v1, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_direct_t, magma_storev_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, trans, direct, storev, m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork, dworkvt, ldworkvt)
end

"""
    magma_zlarfbx_gpu_v1(m, k, V, ldv, dT, ldt, c, dwork)


### Prototype
```c
void magma_zlarfbx_gpu_v1( magma_int_t m, magma_int_t k, magmaDoubleComplex_ptr V, magma_int_t ldv, magmaDoubleComplex_ptr dT, magma_int_t ldt, magmaDoubleComplex_ptr c, magmaDoubleComplex_ptr dwork );
```
"""
function magma_zlarfbx_gpu_v1(m, k, V, ldv, dT, ldt, c, dwork)
    ccall((:magma_zlarfbx_gpu_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr), m, k, V, ldv, dT, ldt, c, dwork)
end

"""
    magma_zlarfg_gpu_v1(n, dx0, dx, dtau, dxnorm, dAkk)


### Prototype
```c
void magma_zlarfg_gpu_v1( magma_int_t n, magmaDoubleComplex_ptr dx0, magmaDoubleComplex_ptr dx, magmaDoubleComplex_ptr dtau, magmaDouble_ptr dxnorm, magmaDoubleComplex_ptr dAkk );
```
"""
function magma_zlarfg_gpu_v1(n, dx0, dx, dtau, dxnorm, dAkk)
    ccall((:magma_zlarfg_gpu_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, magmaDoubleComplex_ptr), n, dx0, dx, dtau, dxnorm, dAkk)
end

"""
    magma_zlarfgtx_gpu_v1(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv, T, ldt, dwork)


### Prototype
```c
void magma_zlarfgtx_gpu_v1( magma_int_t n, magmaDoubleComplex_ptr dx0, magmaDoubleComplex_ptr dx, magmaDoubleComplex_ptr dtau, magmaDouble_ptr dxnorm, magmaDoubleComplex_ptr dA, magma_int_t iter, magmaDoubleComplex_ptr V, magma_int_t ldv, magmaDoubleComplex_ptr T, magma_int_t ldt, magmaDoubleComplex_ptr dwork );
```
"""
function magma_zlarfgtx_gpu_v1(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv, T, ldt, dwork)
    ccall((:magma_zlarfgtx_gpu_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr), n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv, T, ldt, dwork)
end

"""
    magma_zlarfgx_gpu_v1(n, dx0, dx, dtau, dxnorm, dA, iter)


### Prototype
```c
void magma_zlarfgx_gpu_v1( magma_int_t n, magmaDoubleComplex_ptr dx0, magmaDoubleComplex_ptr dx, magmaDoubleComplex_ptr dtau, magmaDouble_ptr dxnorm, magmaDoubleComplex_ptr dA, magma_int_t iter );
```
"""
function magma_zlarfgx_gpu_v1(n, dx0, dx, dtau, dxnorm, dA, iter)
    ccall((:magma_zlarfgx_gpu_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, magmaDoubleComplex_ptr, magma_int_t), n, dx0, dx, dtau, dxnorm, dA, iter)
end

"""
    magma_zlarfx_gpu_v1(m, n, v, tau, C, ldc, xnorm, dT, iter, work)


### Prototype
```c
void magma_zlarfx_gpu_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr v, magmaDoubleComplex_ptr tau, magmaDoubleComplex_ptr C, magma_int_t ldc, magmaDouble_ptr xnorm, magmaDoubleComplex_ptr dT, magma_int_t iter, magmaDoubleComplex_ptr work );
```
"""
function magma_zlarfx_gpu_v1(m, n, v, tau, C, ldc, xnorm, dT, iter, work)
    ccall((:magma_zlarfx_gpu_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDouble_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr), m, n, v, tau, C, ldc, xnorm, dT, iter, work)
end

"""
    magmablas_zaxpycp_v1(m, dr, dx, db)

Level 1 BLAS (alphabetical order)
### Prototype
```c
void magmablas_zaxpycp_v1( magma_int_t m, magmaDoubleComplex_ptr dr, magmaDoubleComplex_ptr dx, magmaDoubleComplex_const_ptr db );
```
"""
function magmablas_zaxpycp_v1(m, dr, dx, db)
    ccall((:magmablas_zaxpycp_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_const_ptr), m, dr, dx, db)
end

"""
    magmablas_zswap_v1(n, dx, incx, dy, incy)


### Prototype
```c
void magmablas_zswap_v1( magma_int_t n, magmaDoubleComplex_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magmablas_zswap_v1(n, dx, incx, dy, incy)
    ccall((:magmablas_zswap_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, dx, incx, dy, incy)
end

@enum magma_order_t::UInt32 begin
    MagmaRowMajor = 101
    MagmaColMajor = 102
end

"""
    magmablas_zswapblk_v1(order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset)


### Prototype
```c
void magmablas_zswapblk_v1( magma_order_t order, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t i1, magma_int_t i2, const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset );
```
"""
function magmablas_zswapblk_v1(order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset)
    ccall((:magmablas_zswapblk_v1, libmagma), Cvoid, (magma_order_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t), order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset)
end

"""
    magmablas_zswapdblk_v1(n, nb, dA, ldda, inca, dB, lddb, incb)


### Prototype
```c
void magmablas_zswapdblk_v1( magma_int_t n, magma_int_t nb, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t inca, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t incb );
```
"""
function magmablas_zswapdblk_v1(n, nb, dA, ldda, inca, dB, lddb, incb)
    ccall((:magmablas_zswapdblk_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t), n, nb, dA, ldda, inca, dB, lddb, incb)
end

"""
    magmablas_zgemv_v1(trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)

Level 2 BLAS (alphabetical order)
### Prototype
```c
void magmablas_zgemv_v1( magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magmablas_zgemv_v1(trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
    ccall((:magmablas_zgemv_v1, libmagma), Cvoid, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
end

"""
    magmablas_zgemv_conj_v1(m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)


### Prototype
```c
void magmablas_zgemv_conj_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magmablas_zgemv_conj_v1(m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
    ccall((:magmablas_zgemv_conj_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
end

"""
    magmablas_zhemv_v1(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)


### Prototype
```c
magma_int_t magmablas_zhemv_v1( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magmablas_zhemv_v1(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
    ccall((:magmablas_zhemv_v1, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
end

"""
    magmablas_zsymv_v1(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)


### Prototype
```c
magma_int_t magmablas_zsymv_v1( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magmablas_zsymv_v1(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
    ccall((:magmablas_zsymv_v1, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
end

"""
    magmablas_zgemm_v1(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)

Level 3 BLAS (alphabetical order)
### Prototype
```c
void magmablas_zgemm_v1( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zgemm_v1(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magmablas_zgemm_v1, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magmablas_zgemm_reduce_v1(m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magmablas_zgemm_reduce_v1( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zgemm_reduce_v1(m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magmablas_zgemm_reduce_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magmablas_zhemm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magmablas_zhemm_v1( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zhemm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magmablas_zhemm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magmablas_zsymm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magmablas_zsymm_v1( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zsymm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magmablas_zsymm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magmablas_zsyr2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magmablas_zsyr2k_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zsyr2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magmablas_zsyr2k_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magmablas_zher2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magmablas_zher2k_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, double beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zher2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magmablas_zher2k_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Cdouble, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magmablas_zsyrk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)


### Prototype
```c
void magmablas_zsyrk_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zsyrk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
    ccall((:magmablas_zsyrk_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
end

"""
    magmablas_zherk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)


### Prototype
```c
void magmablas_zherk_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, double beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magmablas_zherk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
    ccall((:magmablas_zherk_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Cdouble, magmaDoubleComplex_const_ptr, magma_int_t, Cdouble, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
end

"""
    magmablas_ztrsm_v1(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb)


### Prototype
```c
void magmablas_ztrsm_v1( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magmablas_ztrsm_v1(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb)
    ccall((:magmablas_ztrsm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb)
end

"""
    magmablas_ztrsm_outofplace_v1(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, dX, lddx, flag, d_dinvA, dinvA_length)


### Prototype
```c
void magmablas_ztrsm_outofplace_v1( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magma_int_t flag, magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length );
```
"""
function magmablas_ztrsm_outofplace_v1(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, dX, lddx, flag, d_dinvA, dinvA_length)
    ccall((:magmablas_ztrsm_outofplace_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, dX, lddx, flag, d_dinvA, dinvA_length)
end

"""
    magmablas_ztrsm_work_v1(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, dX, lddx, flag, d_dinvA, dinvA_length)


### Prototype
```c
void magmablas_ztrsm_work_v1( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magma_int_t flag, magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length );
```
"""
function magmablas_ztrsm_work_v1(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, dX, lddx, flag, d_dinvA, dinvA_length)
    ccall((:magmablas_ztrsm_work_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, dX, lddx, flag, d_dinvA, dinvA_length)
end

"""
    magma_izamax_v1(n, dx, incx)

in cublas_v2, result returned through output argument
### Prototype
```c
magma_int_t magma_izamax_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx );
```
"""
function magma_izamax_v1(n, dx, incx)
    ccall((:magma_izamax_v1, libmagma), magma_int_t, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t), n, dx, incx)
end

"""
    magma_izamin_v1(n, dx, incx)

in cublas_v2, result returned through output argument
### Prototype
```c
magma_int_t magma_izamin_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx );
```
"""
function magma_izamin_v1(n, dx, incx)
    ccall((:magma_izamin_v1, libmagma), magma_int_t, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t), n, dx, incx)
end

"""
    magma_dzasum_v1(n, dx, incx)

in cublas_v2, result returned through output argument
### Prototype
```c
double magma_dzasum_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx );
```
"""
function magma_dzasum_v1(n, dx, incx)
    ccall((:magma_dzasum_v1, libmagma), Cdouble, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t), n, dx, incx)
end

"""
    magma_zaxpy_v1(n, alpha, dx, incx, dy, incy)


### Prototype
```c
void magma_zaxpy_v1( magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magma_zaxpy_v1(n, alpha, dx, incx, dy, incy)
    ccall((:magma_zaxpy_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, alpha, dx, incx, dy, incy)
end

"""
    magma_zcopy_v1(n, dx, incx, dy, incy)


### Prototype
```c
void magma_zcopy_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magma_zcopy_v1(n, dx, incx, dy, incy)
    ccall((:magma_zcopy_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, dx, incx, dy, incy)
end

"""
    magma_zdotc_v1(n, dx, incx, dy, incy)

in cublas_v2, result returned through output argument
### Prototype
```c
magmaDoubleComplex magma_zdotc_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_const_ptr dy, magma_int_t incy );
```
"""
function magma_zdotc_v1(n, dx, incx, dy, incy)
    ccall((:magma_zdotc_v1, libmagma), magmaDoubleComplex, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t), n, dx, incx, dy, incy)
end

"""
    magma_zdotu_v1(n, dx, incx, dy, incy)

in cublas_v2, result returned through output argument
### Prototype
```c
magmaDoubleComplex magma_zdotu_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_const_ptr dy, magma_int_t incy );
```
"""
function magma_zdotu_v1(n, dx, incx, dy, incy)
    ccall((:magma_zdotu_v1, libmagma), magmaDoubleComplex, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t), n, dx, incx, dy, incy)
end

"""
    magma_dznrm2_v1(n, dx, incx)

in cublas_v2, result returned through output argument
### Prototype
```c
double magma_dznrm2_v1( magma_int_t n, magmaDoubleComplex_const_ptr dx, magma_int_t incx );
```
"""
function magma_dznrm2_v1(n, dx, incx)
    ccall((:magma_dznrm2_v1, libmagma), Cdouble, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t), n, dx, incx)
end

"""
    magma_zrot_v1(n, dx, incx, dy, incy, dc, ds)


### Prototype
```c
void magma_zrot_v1( magma_int_t n, magmaDoubleComplex_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dy, magma_int_t incy, double dc, magmaDoubleComplex ds );
```
"""
function magma_zrot_v1(n, dx, incx, dy, incy, dc, ds)
    ccall((:magma_zrot_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Cdouble, magmaDoubleComplex), n, dx, incx, dy, incy, dc, ds)
end

"""
    magma_zdrot_v1(n, dx, incx, dy, incy, dc, ds)


### Prototype
```c
void magma_zdrot_v1( magma_int_t n, magmaDoubleComplex_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dy, magma_int_t incy, double dc, double ds );
```
"""
function magma_zdrot_v1(n, dx, incx, dy, incy, dc, ds)
    ccall((:magma_zdrot_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Cdouble, Cdouble), n, dx, incx, dy, incy, dc, ds)
end

"""
    magma_zscal_v1(n, alpha, dx, incx)


### Prototype
```c
void magma_zscal_v1( magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dx, magma_int_t incx );
```
"""
function magma_zscal_v1(n, alpha, dx, incx)
    ccall((:magma_zscal_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), n, alpha, dx, incx)
end

"""
    magma_zdscal_v1(n, alpha, dx, incx)


### Prototype
```c
void magma_zdscal_v1( magma_int_t n, double alpha, magmaDoubleComplex_ptr dx, magma_int_t incx );
```
"""
function magma_zdscal_v1(n, alpha, dx, incx)
    ccall((:magma_zdscal_v1, libmagma), Cvoid, (magma_int_t, Cdouble, magmaDoubleComplex_ptr, magma_int_t), n, alpha, dx, incx)
end

"""
    magma_zswap_v1(n, dx, incx, dy, incy)


### Prototype
```c
void magma_zswap_v1( magma_int_t n, magmaDoubleComplex_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magma_zswap_v1(n, dx, incx, dy, incy)
    ccall((:magma_zswap_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), n, dx, incx, dy, incy)
end

"""
    magma_zgemv_v1(transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)

=============================================================================
Level 2 BLAS (alphabetical order)
### Prototype
```c
void magma_zgemv_v1( magma_trans_t transA, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magma_zgemv_v1(transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
    ccall((:magma_zgemv_v1, libmagma), Cvoid, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), transA, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
end

"""
    magma_zgerc_v1(m, n, alpha, dx, incx, dy, incy, dA, ldda)


### Prototype
```c
void magma_zgerc_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_const_ptr dy, magma_int_t incy, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magma_zgerc_v1(m, n, alpha, dx, incx, dy, incy, dA, ldda)
    ccall((:magma_zgerc_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), m, n, alpha, dx, incx, dy, incy, dA, ldda)
end

"""
    magma_zgeru_v1(m, n, alpha, dx, incx, dy, incy, dA, ldda)


### Prototype
```c
void magma_zgeru_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_const_ptr dy, magma_int_t incy, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magma_zgeru_v1(m, n, alpha, dx, incx, dy, incy, dA, ldda)
    ccall((:magma_zgeru_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), m, n, alpha, dx, incx, dy, incy, dA, ldda)
end

"""
    magma_zhemv_v1(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)


### Prototype
```c
void magma_zhemv_v1( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy );
```
"""
function magma_zhemv_v1(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
    ccall((:magma_zhemv_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy)
end

"""
    magma_zher_v1(uplo, n, alpha, dx, incx, dA, ldda)


### Prototype
```c
void magma_zher_v1( magma_uplo_t uplo, magma_int_t n, double alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magma_zher_v1(uplo, n, alpha, dx, incx, dA, ldda)
    ccall((:magma_zher_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, Cdouble, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, n, alpha, dx, incx, dA, ldda)
end

"""
    magma_zher2_v1(uplo, n, alpha, dx, incx, dy, incy, dA, ldda)


### Prototype
```c
void magma_zher2_v1( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_const_ptr dy, magma_int_t incy, magmaDoubleComplex_ptr dA, magma_int_t ldda );
```
"""
function magma_zher2_v1(uplo, n, alpha, dx, incx, dy, incy, dA, ldda)
    ccall((:magma_zher2_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, n, alpha, dx, incx, dy, incy, dA, ldda)
end

"""
    magma_ztrmv_v1(uplo, trans, diag, n, dA, ldda, dx, incx)


### Prototype
```c
void magma_ztrmv_v1( magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dx, magma_int_t incx );
```
"""
function magma_ztrmv_v1(uplo, trans, diag, n, dA, ldda, dx, incx)
    ccall((:magma_ztrmv_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, diag, n, dA, ldda, dx, incx)
end

"""
    magma_ztrsv_v1(uplo, trans, diag, n, dA, ldda, dx, incx)


### Prototype
```c
void magma_ztrsv_v1( magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dx, magma_int_t incx );
```
"""
function magma_ztrsv_v1(uplo, trans, diag, n, dA, ldda, dx, incx)
    ccall((:magma_ztrsv_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, diag, n, dA, ldda, dx, incx)
end

"""
    magma_zgemm_v1(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)

=============================================================================
Level 3 BLAS (alphabetical order)
### Prototype
```c
void magma_zgemm_v1( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zgemm_v1(transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magma_zgemm_v1, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), transA, transB, m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magma_zsymm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magma_zsymm_v1( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zsymm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magma_zsymm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magma_zhemm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magma_zhemm_v1( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zhemm_v1(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magma_zhemm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magma_zsyr2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magma_zsyr2k_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zsyr2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magma_zsyr2k_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magma_zher2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)


### Prototype
```c
void magma_zher2k_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, double beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zher2k_v1(uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
    ccall((:magma_zher2k_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Cdouble, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc)
end

"""
    magma_zsyrk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)


### Prototype
```c
void magma_zsyrk_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zsyrk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
    ccall((:magma_zsyrk_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
end

"""
    magma_zherk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)


### Prototype
```c
void magma_zherk_v1( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, double beta, magmaDoubleComplex_ptr dC, magma_int_t lddc );
```
"""
function magma_zherk_v1(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
    ccall((:magma_zherk_v1, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Cdouble, magmaDoubleComplex_const_ptr, magma_int_t, Cdouble, magmaDoubleComplex_ptr, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc)
end

"""
    magma_ztrmm_v1(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb)


### Prototype
```c
void magma_ztrmm_v1( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magma_ztrmm_v1(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb)
    ccall((:magma_ztrmm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb)
end

"""
    magma_ztrsm_v1(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb)


### Prototype
```c
void magma_ztrsm_v1( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb );
```
"""
function magma_ztrsm_v1(side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb)
    ccall((:magma_ztrsm_v1, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), side, uplo, trans, diag, m, n, alpha, dA, ldda, dB, lddb)
end

const magmaFloatComplex = Cint

const magmaFloatComplex_ptr = Ptr{magmaFloatComplex}

"""
    magmablas_zcaxpycp_v1(m, r, x, b, w)

Mixed precision 
### Prototype
```c
void magmablas_zcaxpycp_v1( magma_int_t m, magmaFloatComplex_ptr r, magmaDoubleComplex_ptr x, magmaDoubleComplex_const_ptr b, magmaDoubleComplex_ptr w );
```
"""
function magmablas_zcaxpycp_v1(m, r, x, b, w)
    ccall((:magmablas_zcaxpycp_v1, libmagma), Cvoid, (magma_int_t, magmaFloatComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_const_ptr, magmaDoubleComplex_ptr), m, r, x, b, w)
end

"""
    magmablas_zclaswp_v1(n, A, lda, SA, m, ipiv, incx)


### Prototype
```c
void magmablas_zclaswp_v1( magma_int_t n, magmaDoubleComplex_ptr A, magma_int_t lda, magmaFloatComplex_ptr SA, magma_int_t m, const magma_int_t *ipiv, magma_int_t incx );
```
"""
function magmablas_zclaswp_v1(n, A, lda, SA, m, ipiv, incx)
    ccall((:magmablas_zclaswp_v1, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaFloatComplex_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t), n, A, lda, SA, m, ipiv, incx)
end

"""
    magmablas_zlag2c_v1(m, n, A, lda, SA, ldsa, info)


### Prototype
```c
void magmablas_zlag2c_v1( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr A, magma_int_t lda, magmaFloatComplex_ptr SA, magma_int_t ldsa, magma_int_t *info );
```
"""
function magmablas_zlag2c_v1(m, n, A, lda, SA, ldsa, info)
    ccall((:magmablas_zlag2c_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaFloatComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, SA, ldsa, info)
end

const magmaFloatComplex_const_ptr = Ptr{magmaFloatComplex}

"""
    magmablas_clag2z_v1(m, n, SA, ldsa, A, lda, info)


### Prototype
```c
void magmablas_clag2z_v1( magma_int_t m, magma_int_t n, magmaFloatComplex_const_ptr SA, magma_int_t ldsa, magmaDoubleComplex_ptr A, magma_int_t lda, magma_int_t *info );
```
"""
function magmablas_clag2z_v1(m, n, SA, ldsa, A, lda, info)
    ccall((:magmablas_clag2z_v1, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaFloatComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, SA, ldsa, A, lda, info)
end

"""
    magmablas_zlat2c_v1(uplo, n, A, lda, SA, ldsa, info)


### Prototype
```c
void magmablas_zlat2c_v1( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_const_ptr A, magma_int_t lda, magmaFloatComplex_ptr SA, magma_int_t ldsa, magma_int_t *info );
```
"""
function magmablas_zlat2c_v1(uplo, n, A, lda, SA, ldsa, info)
    ccall((:magmablas_zlat2c_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaFloatComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, A, lda, SA, ldsa, info)
end

"""
    magmablas_clat2z_v1(uplo, n, SA, ldsa, A, lda, info)


### Prototype
```c
void magmablas_clat2z_v1( magma_uplo_t uplo, magma_int_t n, magmaFloatComplex_const_ptr SA, magma_int_t ldsa, magmaDoubleComplex_ptr A, magma_int_t lda, magma_int_t *info );
```
"""
function magmablas_clat2z_v1(uplo, n, SA, ldsa, A, lda, info)
    ccall((:magmablas_clat2z_v1, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaFloatComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, SA, ldsa, A, lda, info)
end

"""
Define new type that the precision generator will not change (matches PLASMA)
"""
const real_Double_t = Cdouble

const magma_event_t = Cint

"""
use short for cuda older than 7.5
corresponding routines would not work anyway since there is no half precision
"""
const magmaHalf = Cshort

"""
    magma_queue_get_cuda_stream(queue)


### Prototype
```c
cudaStream_t magma_queue_get_cuda_stream ( magma_queue_t queue );
```
"""
function magma_queue_get_cuda_stream(queue)
    ccall((:magma_queue_get_cuda_stream, libmagma), Cint, (magma_queue_t,), queue)
end

"""
    magma_queue_get_cublas_handle(queue)


### Prototype
```c
cublasHandle_t magma_queue_get_cublas_handle ( magma_queue_t queue );
```
"""
function magma_queue_get_cublas_handle(queue)
    ccall((:magma_queue_get_cublas_handle, libmagma), Cint, (magma_queue_t,), queue)
end

"""
    magma_queue_get_cusparse_handle(queue)


### Prototype
```c
cusparseHandle_t magma_queue_get_cusparse_handle( magma_queue_t queue );
```
"""
function magma_queue_get_cusparse_handle(queue)
    ccall((:magma_queue_get_cusparse_handle, libmagma), Cint, (magma_queue_t,), queue)
end

"""
    magma_cabs(x)

for MAGMA_[CZ]_ABS
### Prototype
```c
double magma_cabs ( magmaDoubleComplex x );
```
"""
function magma_cabs(x)
    ccall((:magma_cabs, libmagma), Cdouble, (magmaDoubleComplex,), x)
end

"""
    magma_cabsf(x)


### Prototype
```c
float magma_cabsf( magmaFloatComplex x );
```
"""
function magma_cabsf(x)
    ccall((:magma_cabsf, libmagma), Cfloat, (magmaFloatComplex,), x)
end

const magmaFloat_ptr = Ptr{Cfloat}

const magmaHalf_ptr = Ptr{magmaHalf}

const magmaFloat_const_ptr = Ptr{Cfloat}

const magmaHalf_const_ptr = Ptr{magmaHalf}

"""
    magma_bool_t

-----------------------------------------------------------------------------
parameter constants
numbering is consistent with CBLAS and PLASMA; see plasma/include/plasma.h
also with lapack_cwrapper/include/lapack_enum.h
see http://www.netlib.org/lapack/lapwrapc/
"""
@enum magma_bool_t::UInt32 begin
    MagmaFalse = 0
    MagmaTrue = 1
end

@enum magma_dist_t::UInt32 begin
    MagmaDistUniform = 201
    MagmaDistSymmetric = 202
    MagmaDistNormal = 203
end

@enum magma_sym_t::UInt32 begin
    MagmaHermGeev = 241
    MagmaHermPoev = 242
    MagmaNonsymPosv = 243
    MagmaSymPosv = 244
end

@enum magma_pack_t::UInt32 begin
    MagmaNoPacking = 291
    MagmaPackSubdiag = 292
    MagmaPackSupdiag = 293
    MagmaPackColumn = 294
    MagmaPackRow = 295
    MagmaPackLowerBand = 296
    MagmaPackUpeprBand = 297
    MagmaPackAll = 298
end

@enum magma_vec_t::UInt32 begin
    MagmaNoVec = 301
    MagmaVec = 302
    MagmaIVec = 303
    MagmaAllVec = 304
    MagmaSomeVec = 305
    MagmaOverwriteVec = 306
    MagmaBacktransVec = 307
end

@enum magma_range_t::UInt32 begin
    MagmaRangeAll = 311
    MagmaRangeV = 312
    MagmaRangeI = 313
end

@enum magma_vect_t::UInt32 begin
    MagmaQ = 322
    MagmaP = 323
end

@enum magma_mode_t::UInt32 begin
    MagmaHybrid = 701
    MagmaNative = 702
end

"""
    magma_storage_t

-----------------------------------------------------------------------------
sparse
"""
@enum magma_storage_t::UInt32 begin
    Magma_CSR = 611
    Magma_ELLPACKT = 612
    Magma_ELL = 613
    Magma_DENSE = 614
    Magma_BCSR = 615
    Magma_CSC = 616
    Magma_HYB = 617
    Magma_COO = 618
    Magma_ELLRT = 619
    Magma_SPMVFUNCTION = 620
    Magma_SELLP = 621
    Magma_ELLD = 622
    Magma_CSRLIST = 623
    Magma_CSRD = 624
    Magma_CSRL = 627
    Magma_CSRU = 628
    Magma_CSRCOO = 629
    Magma_CUCSR = 630
    Magma_COOLIST = 631
    Magma_CSR5 = 632
end

@enum magma_solver_type::UInt32 begin
    Magma_CG = 431
    Magma_CGMERGE = 432
    Magma_GMRES = 433
    Magma_BICGSTAB = 434
    Magma_BICGSTABMERGE = 435
    Magma_BICGSTABMERGE2 = 436
    Magma_JACOBI = 437
    Magma_GS = 438
    Magma_ITERREF = 439
    Magma_BCSRLU = 440
    Magma_PCG = 441
    Magma_PGMRES = 442
    Magma_PBICGSTAB = 443
    Magma_PASTIX = 444
    Magma_ILU = 445
    Magma_ICC = 446
    Magma_PARILU = 447
    Magma_PARIC = 448
    Magma_BAITER = 449
    Magma_LOBPCG = 450
    Magma_NONE = 451
    Magma_FUNCTION = 452
    Magma_IDR = 453
    Magma_PIDR = 454
    Magma_CGS = 455
    Magma_PCGS = 456
    Magma_CGSMERGE = 457
    Magma_PCGSMERGE = 458
    Magma_TFQMR = 459
    Magma_PTFQMR = 460
    Magma_TFQMRMERGE = 461
    Magma_PTFQMRMERGE = 462
    Magma_QMR = 463
    Magma_PQMR = 464
    Magma_QMRMERGE = 465
    Magma_PQMRMERGE = 466
    Magma_BOMBARD = 490
    Magma_BOMBARDMERGE = 491
    Magma_PCGMERGE = 492
    Magma_BAITERO = 493
    Magma_IDRMERGE = 494
    Magma_PBICGSTABMERGE = 495
    Magma_PARICT = 496
    Magma_CUSTOMIC = 497
    Magma_CUSTOMILU = 498
    Magma_PIDRMERGE = 499
    Magma_BICG = 500
    Magma_BICGMERGE = 501
    Magma_PBICG = 502
    Magma_PBICGMERGE = 503
    Magma_LSQR = 504
    Magma_PARILUT = 505
    Magma_ISAI = 506
    Magma_CUSOLVE = 507
    Magma_VBJACOBI = 508
    Magma_PARDISO = 509
    Magma_SYNCFREESOLVE = 510
    Magma_ILUT = 511
end

@enum magma_ortho_t::UInt32 begin
    Magma_CGSO = 561
    Magma_FUSED_CGSO = 562
    Magma_MGSO = 563
end

@enum magma_location_t::UInt32 begin
    Magma_CPU = 571
    Magma_DEV = 572
end

@enum magma_symmetry_t::UInt32 begin
    Magma_GENERAL = 581
    Magma_SYMMETRIC = 582
end

@enum magma_diagorder_t::UInt32 begin
    Magma_ORDERED = 591
    Magma_DIAGFIRST = 592
    Magma_UNITY = 593
    Magma_VALUE = 594
end

@enum magma_precision::UInt32 begin
    Magma_DCOMPLEX = 501
    Magma_FCOMPLEX = 502
    Magma_DOUBLE = 503
    Magma_FLOAT = 504
end

@enum magma_scale_t::UInt32 begin
    Magma_NOSCALE = 511
    Magma_UNITROW = 512
    Magma_UNITDIAG = 513
    Magma_UNITCOL = 514
    Magma_UNITROWCOL = 515
    Magma_UNITDIAGCOL = 516
end

@enum magma_operation_t::UInt32 begin
    Magma_SOLVE = 801
    Magma_SETUPSOLVE = 802
    Magma_APPLYSOLVE = 803
    Magma_DESTROYSOLVE = 804
    Magma_INFOSOLVE = 805
    Magma_GENERATEPREC = 806
    Magma_PRECONDLEFT = 807
    Magma_PRECONDRIGHT = 808
    Magma_TRANSPOSE = 809
    Magma_SPMV = 810
end

@enum magma_refinement_t::UInt32 begin
    Magma_PREC_SS = 900
    Magma_PREC_SST = 901
    Magma_PREC_HS = 902
    Magma_PREC_HST = 903
    Magma_PREC_SH = 904
    Magma_PREC_SHT = 905
    Magma_PREC_XHS_H = 910
    Magma_PREC_XHS_HTC = 911
    Magma_PREC_XHS_161616 = 912
    Magma_PREC_XHS_161616TC = 913
    Magma_PREC_XHS_161632TC = 914
    Magma_PREC_XSH_S = 915
    Magma_PREC_XSH_STC = 916
    Magma_PREC_XSH_163232TC = 917
    Magma_PREC_XSH_323232TC = 918
    Magma_REFINE_IRSTRS = 920
    Magma_REFINE_IRDTRS = 921
    Magma_REFINE_IRGMSTRS = 922
    Magma_REFINE_IRGMDTRS = 923
    Magma_REFINE_GMSTRS = 924
    Magma_REFINE_GMDTRS = 925
    Magma_REFINE_GMGMSTRS = 926
    Magma_REFINE_GMGMDTRS = 927
    Magma_PREC_HD = 930
end

@enum magma_mp_type_t::UInt32 begin
    Magma_MP_BASE_SS = 950
    Magma_MP_BASE_DD = 951
    Magma_MP_BASE_XHS = 952
    Magma_MP_BASE_XSH = 953
    Magma_MP_BASE_XHD = 954
    Magma_MP_BASE_XDH = 955
    Magma_MP_ENABLE_DFLT_MATH = 960
    Magma_MP_ENABLE_TC_MATH = 961
    Magma_MP_SGEMM = 962
    Magma_MP_HGEMM = 963
    Magma_MP_GEMEX_I32_O32_C32 = 964
    Magma_MP_GEMEX_I16_O32_C32 = 965
    Magma_MP_GEMEX_I16_O16_C32 = 966
    Magma_MP_GEMEX_I16_O16_C16 = 967
    Magma_MP_TC_SGEMM = 968
    Magma_MP_TC_HGEMM = 969
    Magma_MP_TC_GEMEX_I32_O32_C32 = 970
    Magma_MP_TC_GEMEX_I16_O32_C32 = 971
    Magma_MP_TC_GEMEX_I16_O16_C32 = 972
    Magma_MP_TC_GEMEX_I16_O16_C16 = 973
end

"""
    magma_bool_const(lapack_char)

-----------------------------------------------------------------------------
Convert LAPACK character constants to MAGMA constants.
This is a one-to-many mapping, requiring multiple translators
(e.g., "N" can be NoTrans or NonUnit or NoVec).
### Prototype
```c
magma_bool_t magma_bool_const ( char lapack_char );
```
"""
function magma_bool_const(lapack_char)
    ccall((:magma_bool_const, libmagma), magma_bool_t, (Cchar,), lapack_char)
end

"""
    magma_order_const(lapack_char)


### Prototype
```c
magma_order_t magma_order_const ( char lapack_char );
```
"""
function magma_order_const(lapack_char)
    ccall((:magma_order_const, libmagma), magma_order_t, (Cchar,), lapack_char)
end

"""
    magma_trans_const(lapack_char)


### Prototype
```c
magma_trans_t magma_trans_const ( char lapack_char );
```
"""
function magma_trans_const(lapack_char)
    ccall((:magma_trans_const, libmagma), magma_trans_t, (Cchar,), lapack_char)
end

"""
    magma_uplo_const(lapack_char)


### Prototype
```c
magma_uplo_t magma_uplo_const ( char lapack_char );
```
"""
function magma_uplo_const(lapack_char)
    ccall((:magma_uplo_const, libmagma), magma_uplo_t, (Cchar,), lapack_char)
end

"""
    magma_diag_const(lapack_char)


### Prototype
```c
magma_diag_t magma_diag_const ( char lapack_char );
```
"""
function magma_diag_const(lapack_char)
    ccall((:magma_diag_const, libmagma), magma_diag_t, (Cchar,), lapack_char)
end

"""
    magma_side_const(lapack_char)


### Prototype
```c
magma_side_t magma_side_const ( char lapack_char );
```
"""
function magma_side_const(lapack_char)
    ccall((:magma_side_const, libmagma), magma_side_t, (Cchar,), lapack_char)
end

"""
    magma_norm_const(lapack_char)


### Prototype
```c
magma_norm_t magma_norm_const ( char lapack_char );
```
"""
function magma_norm_const(lapack_char)
    ccall((:magma_norm_const, libmagma), magma_norm_t, (Cchar,), lapack_char)
end

"""
    magma_dist_const(lapack_char)


### Prototype
```c
magma_dist_t magma_dist_const ( char lapack_char );
```
"""
function magma_dist_const(lapack_char)
    ccall((:magma_dist_const, libmagma), magma_dist_t, (Cchar,), lapack_char)
end

"""
    magma_sym_const(lapack_char)


### Prototype
```c
magma_sym_t magma_sym_const ( char lapack_char );
```
"""
function magma_sym_const(lapack_char)
    ccall((:magma_sym_const, libmagma), magma_sym_t, (Cchar,), lapack_char)
end

"""
    magma_pack_const(lapack_char)


### Prototype
```c
magma_pack_t magma_pack_const ( char lapack_char );
```
"""
function magma_pack_const(lapack_char)
    ccall((:magma_pack_const, libmagma), magma_pack_t, (Cchar,), lapack_char)
end

"""
    magma_vec_const(lapack_char)


### Prototype
```c
magma_vec_t magma_vec_const ( char lapack_char );
```
"""
function magma_vec_const(lapack_char)
    ccall((:magma_vec_const, libmagma), magma_vec_t, (Cchar,), lapack_char)
end

"""
    magma_range_const(lapack_char)


### Prototype
```c
magma_range_t magma_range_const ( char lapack_char );
```
"""
function magma_range_const(lapack_char)
    ccall((:magma_range_const, libmagma), magma_range_t, (Cchar,), lapack_char)
end

"""
    magma_vect_const(lapack_char)


### Prototype
```c
magma_vect_t magma_vect_const ( char lapack_char );
```
"""
function magma_vect_const(lapack_char)
    ccall((:magma_vect_const, libmagma), magma_vect_t, (Cchar,), lapack_char)
end

"""
    magma_direct_const(lapack_char)


### Prototype
```c
magma_direct_t magma_direct_const( char lapack_char );
```
"""
function magma_direct_const(lapack_char)
    ccall((:magma_direct_const, libmagma), magma_direct_t, (Cchar,), lapack_char)
end

"""
    magma_storev_const(lapack_char)


### Prototype
```c
magma_storev_t magma_storev_const( char lapack_char );
```
"""
function magma_storev_const(lapack_char)
    ccall((:magma_storev_const, libmagma), magma_storev_t, (Cchar,), lapack_char)
end

"""
    lapack_const_str(magma_const)

magma  defines lapack_const_str, which returns char* to call lapack (Fortran interface).
plasma defines lapack_const, which is roughly the same as MAGMA's lapacke_const
(returns a char instead of char*) to call lapacke (C interface).
### Prototype
```c
const char* lapack_const_str ( int magma_const );
```
"""
function lapack_const_str(magma_const)
    ccall((:lapack_const_str, libmagma), Ptr{Cchar}, (Cint,), magma_const)
end

"""
    lapack_bool_const(magma_const)


### Prototype
```c
const char* lapack_bool_const ( magma_bool_t magma_const );
```
"""
function lapack_bool_const(magma_const)
    ccall((:lapack_bool_const, libmagma), Ptr{Cchar}, (magma_bool_t,), magma_const)
end

"""
    lapack_order_const(magma_const)


### Prototype
```c
const char* lapack_order_const ( magma_order_t magma_const );
```
"""
function lapack_order_const(magma_const)
    ccall((:lapack_order_const, libmagma), Ptr{Cchar}, (magma_order_t,), magma_const)
end

"""
    lapack_trans_const(magma_const)


### Prototype
```c
const char* lapack_trans_const ( magma_trans_t magma_const );
```
"""
function lapack_trans_const(magma_const)
    ccall((:lapack_trans_const, libmagma), Ptr{Cchar}, (magma_trans_t,), magma_const)
end

"""
    lapack_uplo_const(magma_const)


### Prototype
```c
const char* lapack_uplo_const ( magma_uplo_t magma_const );
```
"""
function lapack_uplo_const(magma_const)
    ccall((:lapack_uplo_const, libmagma), Ptr{Cchar}, (magma_uplo_t,), magma_const)
end

"""
    lapack_diag_const(magma_const)


### Prototype
```c
const char* lapack_diag_const ( magma_diag_t magma_const );
```
"""
function lapack_diag_const(magma_const)
    ccall((:lapack_diag_const, libmagma), Ptr{Cchar}, (magma_diag_t,), magma_const)
end

"""
    lapack_side_const(magma_const)


### Prototype
```c
const char* lapack_side_const ( magma_side_t magma_const );
```
"""
function lapack_side_const(magma_const)
    ccall((:lapack_side_const, libmagma), Ptr{Cchar}, (magma_side_t,), magma_const)
end

"""
    lapack_norm_const(magma_const)


### Prototype
```c
const char* lapack_norm_const ( magma_norm_t magma_const );
```
"""
function lapack_norm_const(magma_const)
    ccall((:lapack_norm_const, libmagma), Ptr{Cchar}, (magma_norm_t,), magma_const)
end

"""
    lapack_dist_const(magma_const)


### Prototype
```c
const char* lapack_dist_const ( magma_dist_t magma_const );
```
"""
function lapack_dist_const(magma_const)
    ccall((:lapack_dist_const, libmagma), Ptr{Cchar}, (magma_dist_t,), magma_const)
end

"""
    lapack_sym_const(magma_const)


### Prototype
```c
const char* lapack_sym_const ( magma_sym_t magma_const );
```
"""
function lapack_sym_const(magma_const)
    ccall((:lapack_sym_const, libmagma), Ptr{Cchar}, (magma_sym_t,), magma_const)
end

"""
    lapack_pack_const(magma_const)


### Prototype
```c
const char* lapack_pack_const ( magma_pack_t magma_const );
```
"""
function lapack_pack_const(magma_const)
    ccall((:lapack_pack_const, libmagma), Ptr{Cchar}, (magma_pack_t,), magma_const)
end

"""
    lapack_vec_const(magma_const)


### Prototype
```c
const char* lapack_vec_const ( magma_vec_t magma_const );
```
"""
function lapack_vec_const(magma_const)
    ccall((:lapack_vec_const, libmagma), Ptr{Cchar}, (magma_vec_t,), magma_const)
end

"""
    lapack_range_const(magma_const)


### Prototype
```c
const char* lapack_range_const ( magma_range_t magma_const );
```
"""
function lapack_range_const(magma_const)
    ccall((:lapack_range_const, libmagma), Ptr{Cchar}, (magma_range_t,), magma_const)
end

"""
    lapack_vect_const(magma_const)


### Prototype
```c
const char* lapack_vect_const ( magma_vect_t magma_const );
```
"""
function lapack_vect_const(magma_const)
    ccall((:lapack_vect_const, libmagma), Ptr{Cchar}, (magma_vect_t,), magma_const)
end

"""
    lapack_direct_const(magma_const)


### Prototype
```c
const char* lapack_direct_const( magma_direct_t magma_const );
```
"""
function lapack_direct_const(magma_const)
    ccall((:lapack_direct_const, libmagma), Ptr{Cchar}, (magma_direct_t,), magma_const)
end

"""
    lapack_storev_const(magma_const)


### Prototype
```c
const char* lapack_storev_const( magma_storev_t magma_const );
```
"""
function lapack_storev_const(magma_const)
    ccall((:lapack_storev_const, libmagma), Ptr{Cchar}, (magma_storev_t,), magma_const)
end

"""
    lapacke_const(magma_const)


### Prototype
```c
static inline char lapacke_const ( int magma_const );
```
"""
function lapacke_const(magma_const)
    ccall((:lapacke_const, libmagma), Cchar, (Cint,), magma_const)
end

"""
    lapacke_bool_const(magma_const)


### Prototype
```c
static inline char lapacke_bool_const ( magma_bool_t magma_const );
```
"""
function lapacke_bool_const(magma_const)
    ccall((:lapacke_bool_const, libmagma), Cchar, (magma_bool_t,), magma_const)
end

"""
    lapacke_order_const(magma_const)


### Prototype
```c
static inline char lapacke_order_const ( magma_order_t magma_const );
```
"""
function lapacke_order_const(magma_const)
    ccall((:lapacke_order_const, libmagma), Cchar, (magma_order_t,), magma_const)
end

"""
    lapacke_trans_const(magma_const)


### Prototype
```c
static inline char lapacke_trans_const ( magma_trans_t magma_const );
```
"""
function lapacke_trans_const(magma_const)
    ccall((:lapacke_trans_const, libmagma), Cchar, (magma_trans_t,), magma_const)
end

"""
    lapacke_uplo_const(magma_const)


### Prototype
```c
static inline char lapacke_uplo_const ( magma_uplo_t magma_const );
```
"""
function lapacke_uplo_const(magma_const)
    ccall((:lapacke_uplo_const, libmagma), Cchar, (magma_uplo_t,), magma_const)
end

"""
    lapacke_diag_const(magma_const)


### Prototype
```c
static inline char lapacke_diag_const ( magma_diag_t magma_const );
```
"""
function lapacke_diag_const(magma_const)
    ccall((:lapacke_diag_const, libmagma), Cchar, (magma_diag_t,), magma_const)
end

"""
    lapacke_side_const(magma_const)


### Prototype
```c
static inline char lapacke_side_const ( magma_side_t magma_const );
```
"""
function lapacke_side_const(magma_const)
    ccall((:lapacke_side_const, libmagma), Cchar, (magma_side_t,), magma_const)
end

"""
    lapacke_norm_const(magma_const)


### Prototype
```c
static inline char lapacke_norm_const ( magma_norm_t magma_const );
```
"""
function lapacke_norm_const(magma_const)
    ccall((:lapacke_norm_const, libmagma), Cchar, (magma_norm_t,), magma_const)
end

"""
    lapacke_dist_const(magma_const)


### Prototype
```c
static inline char lapacke_dist_const ( magma_dist_t magma_const );
```
"""
function lapacke_dist_const(magma_const)
    ccall((:lapacke_dist_const, libmagma), Cchar, (magma_dist_t,), magma_const)
end

"""
    lapacke_sym_const(magma_const)


### Prototype
```c
static inline char lapacke_sym_const ( magma_sym_t magma_const );
```
"""
function lapacke_sym_const(magma_const)
    ccall((:lapacke_sym_const, libmagma), Cchar, (magma_sym_t,), magma_const)
end

"""
    lapacke_pack_const(magma_const)


### Prototype
```c
static inline char lapacke_pack_const ( magma_pack_t magma_const );
```
"""
function lapacke_pack_const(magma_const)
    ccall((:lapacke_pack_const, libmagma), Cchar, (magma_pack_t,), magma_const)
end

"""
    lapacke_vec_const(magma_const)


### Prototype
```c
static inline char lapacke_vec_const ( magma_vec_t magma_const );
```
"""
function lapacke_vec_const(magma_const)
    ccall((:lapacke_vec_const, libmagma), Cchar, (magma_vec_t,), magma_const)
end

"""
    lapacke_range_const(magma_const)


### Prototype
```c
static inline char lapacke_range_const ( magma_range_t magma_const );
```
"""
function lapacke_range_const(magma_const)
    ccall((:lapacke_range_const, libmagma), Cchar, (magma_range_t,), magma_const)
end

"""
    lapacke_vect_const(magma_const)


### Prototype
```c
static inline char lapacke_vect_const ( magma_vect_t magma_const );
```
"""
function lapacke_vect_const(magma_const)
    ccall((:lapacke_vect_const, libmagma), Cchar, (magma_vect_t,), magma_const)
end

"""
    lapacke_direct_const(magma_const)


### Prototype
```c
static inline char lapacke_direct_const( magma_direct_t magma_const );
```
"""
function lapacke_direct_const(magma_const)
    ccall((:lapacke_direct_const, libmagma), Cchar, (magma_direct_t,), magma_const)
end

"""
    lapacke_storev_const(magma_const)


### Prototype
```c
static inline char lapacke_storev_const( magma_storev_t magma_const );
```
"""
function lapacke_storev_const(magma_const)
    ccall((:lapacke_storev_const, libmagma), Cchar, (magma_storev_t,), magma_const)
end

"""
    magmablas_zgetmatrix_transpose_mgpu(ngpu, m, n, nb, dAT, ldda, hA, lda, dwork, lddw, queues)


### Prototype
```c
void magmablas_zgetmatrix_transpose_mgpu( magma_int_t ngpu, magma_int_t m, magma_int_t n, magma_int_t nb, magmaDoubleComplex_const_ptr const dAT[], magma_int_t ldda, magmaDoubleComplex *hA, magma_int_t lda, magmaDoubleComplex_ptr dwork[], magma_int_t lddw, magma_queue_t queues[][2] );
```
"""
function magmablas_zgetmatrix_transpose_mgpu(ngpu, m, n, nb, dAT, ldda, hA, lda, dwork, lddw, queues)
    ccall((:magmablas_zgetmatrix_transpose_mgpu, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{NTuple{2, magma_queue_t}}), ngpu, m, n, nb, dAT, ldda, hA, lda, dwork, lddw, queues)
end

"""
    magmablas_zsetmatrix_transpose_mgpu(ngpu, m, n, nb, hA, lda, dAT, ldda, dwork, lddw, queues)


### Prototype
```c
void magmablas_zsetmatrix_transpose_mgpu( magma_int_t ngpu, magma_int_t m, magma_int_t n, magma_int_t nb, const magmaDoubleComplex *hA, magma_int_t lda, magmaDoubleComplex_ptr dAT[], magma_int_t ldda, magmaDoubleComplex_ptr dwork[], magma_int_t lddw, magma_queue_t queues[][2] );
```
"""
function magmablas_zsetmatrix_transpose_mgpu(ngpu, m, n, nb, hA, lda, dAT, ldda, dwork, lddw, queues)
    ccall((:magmablas_zsetmatrix_transpose_mgpu, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{NTuple{2, magma_queue_t}}), ngpu, m, n, nb, hA, lda, dAT, ldda, dwork, lddw, queues)
end

"""
    magma_zhtodhe(ngpu, uplo, n, nb, A, lda, dA, ldda, queues, info)

in src/zhetrd_mgpu.cpp
TODO rename zsetmatrix_sy or similar
### Prototype
```c
magma_int_t magma_zhtodhe( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_queue_t queues[][10], magma_int_t *info );
```
"""
function magma_zhtodhe(ngpu, uplo, n, nb, A, lda, dA, ldda, queues, info)
    ccall((:magma_zhtodhe, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{NTuple{10, magma_queue_t}}, Ptr{magma_int_t}), ngpu, uplo, n, nb, A, lda, dA, ldda, queues, info)
end

"""
    magma_zhtodpo(ngpu, uplo, m, n, off_i, off_j, nb, A, lda, dA, ldda, queues, info)

in src/zpotrf3_mgpu.cpp
TODO same as magma_zhtodhe?
### Prototype
```c
magma_int_t magma_zhtodpo( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_queue_t queues[][3], magma_int_t *info );
```
"""
function magma_zhtodpo(ngpu, uplo, m, n, off_i, off_j, nb, A, lda, dA, ldda, queues, info)
    ccall((:magma_zhtodpo, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{NTuple{3, magma_queue_t}}, Ptr{magma_int_t}), ngpu, uplo, m, n, off_i, off_j, nb, A, lda, dA, ldda, queues, info)
end

"""
    magma_zdtohpo(ngpu, uplo, m, n, off_i, off_j, nb, NB, A, lda, dA, ldda, queues, info)

in src/zpotrf3_mgpu.cpp
TODO rename zgetmatrix_sy or similar
### Prototype
```c
magma_int_t magma_zdtohpo( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_queue_t queues[][3], magma_int_t *info );
```
"""
function magma_zdtohpo(ngpu, uplo, m, n, off_i, off_j, nb, NB, A, lda, dA, ldda, queues, info)
    ccall((:magma_zdtohpo, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{NTuple{3, magma_queue_t}}, Ptr{magma_int_t}), ngpu, uplo, m, n, off_i, off_j, nb, NB, A, lda, dA, ldda, queues, info)
end

"""
    magmablas_zhemm_mgpu(side, uplo, m, n, alpha, dA, ldda, offset, dB, lddb, beta, dC, lddc, dwork, dworksiz, ngpu, nb, queues, nqueue, events, nevents, gnode, ncmplx)

Multi-GPU BLAS functions (alphabetical order)
### Prototype
```c
void magmablas_zhemm_mgpu( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t offset, magmaDoubleComplex_ptr dB[], magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC[], magma_int_t lddc, magmaDoubleComplex_ptr dwork[], magma_int_t dworksiz, magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[][20], magma_int_t nqueue, magma_event_t events[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nevents, magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t ncmplx );
```
"""
function magmablas_zhemm_mgpu(side, uplo, m, n, alpha, dA, ldda, offset, dB, lddb, beta, dC, lddc, dwork, dworksiz, ngpu, nb, queues, nqueue, events, nevents, gnode, ncmplx)
    ccall((:magmablas_zhemm_mgpu, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{NTuple{20, magma_queue_t}}, magma_int_t, Ptr{NTuple{74, magma_event_t}}, magma_int_t, Ptr{NTuple{10, magma_int_t}}, magma_int_t), side, uplo, m, n, alpha, dA, ldda, offset, dB, lddb, beta, dC, lddc, dwork, dworksiz, ngpu, nb, queues, nqueue, events, nevents, gnode, ncmplx)
end

"""
    magmablas_zhemv_mgpu(uplo, n, alpha, d_lA, ldda, offset, dx, incx, beta, dy, incy, hwork, lhwork, dwork, ldwork, ngpu, nb, queues)


### Prototype
```c
magma_int_t magmablas_zhemv_mgpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy, magmaDoubleComplex *hwork, magma_int_t lhwork, magmaDoubleComplex_ptr dwork[], magma_int_t ldwork, magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[] );
```
"""
function magmablas_zhemv_mgpu(uplo, n, alpha, d_lA, ldda, offset, dx, incx, beta, dy, incy, hwork, lhwork, dwork, ldwork, ngpu, nb, queues)
    ccall((:magmablas_zhemv_mgpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_queue_t}), uplo, n, alpha, d_lA, ldda, offset, dx, incx, beta, dy, incy, hwork, lhwork, dwork, ldwork, ngpu, nb, queues)
end

"""
    magmablas_zhemv_mgpu_sync(uplo, n, alpha, d_lA, ldda, offset, dx, incx, beta, dy, incy, hwork, lhwork, dwork, ldwork, ngpu, nb, queues)


### Prototype
```c
magma_int_t magmablas_zhemv_mgpu_sync( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy, magmaDoubleComplex *hwork, magma_int_t lhwork, magmaDoubleComplex_ptr dwork[], magma_int_t ldwork, magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[] );
```
"""
function magmablas_zhemv_mgpu_sync(uplo, n, alpha, d_lA, ldda, offset, dx, incx, beta, dy, incy, hwork, lhwork, dwork, ldwork, ngpu, nb, queues)
    ccall((:magmablas_zhemv_mgpu_sync, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_queue_t}), uplo, n, alpha, d_lA, ldda, offset, dx, incx, beta, dy, incy, hwork, lhwork, dwork, ldwork, ngpu, nb, queues)
end

"""
    magma_zhetrs_gpu(uplo, n, nrhs, dA, ldda, ipiv, dB, lddb, info, queue)


### Prototype
```c
magma_int_t magma_zhetrs_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *ipiv, magmaDoubleComplex *dB, magma_int_t lddb, magma_int_t *info, magma_queue_t queue );
```
"""
function magma_zhetrs_gpu(uplo, n, nrhs, dA, ldda, ipiv, dB, lddb, info, queue)
    ccall((:magma_zhetrs_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), uplo, n, nrhs, dA, ldda, ipiv, dB, lddb, info, queue)
end

"""
    magma_zher2k_mgpu(ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)

Ichi's version, in src/zhetrd_mgpu.cpp
### Prototype
```c
void magma_zher2k_mgpu( magma_int_t ngpu, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset, double beta, magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset, magma_int_t nqueue, magma_queue_t queues[][10] );
```
"""
function magma_zher2k_mgpu(ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)
    ccall((:magma_zher2k_mgpu, libmagma), Cvoid, (magma_int_t, magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{NTuple{10, magma_queue_t}}), ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)
end

"""
    magmablas_zher2k_mgpu2(uplo, trans, n, k, alpha, dA, ldda, a_offset, dB, lddb, b_offset, beta, dC, lddc, c_offset, ngpu, nb, queues, nqueue)


### Prototype
```c
void magmablas_zher2k_mgpu2( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t a_offset, magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset, double beta, magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset, magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[][20], magma_int_t nqueue );
```
"""
function magmablas_zher2k_mgpu2(uplo, trans, n, k, alpha, dA, ldda, a_offset, dB, lddb, b_offset, beta, dC, lddc, c_offset, ngpu, nb, queues, nqueue)
    ccall((:magmablas_zher2k_mgpu2, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{NTuple{20, magma_queue_t}}, magma_int_t), uplo, trans, n, k, alpha, dA, ldda, a_offset, dB, lddb, b_offset, beta, dC, lddc, c_offset, ngpu, nb, queues, nqueue)
end

"""
    magma_zherk_mgpu(ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)

in src/zpotrf_mgpu_right.cpp
### Prototype
```c
void magma_zherk_mgpu( magma_int_t ngpu, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset, double beta, magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset, magma_int_t nqueue, magma_queue_t queues[][10] );
```
"""
function magma_zherk_mgpu(ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)
    ccall((:magma_zherk_mgpu, libmagma), Cvoid, (magma_int_t, magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{NTuple{10, magma_queue_t}}), ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)
end

"""
    magma_zherk_mgpu2(ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)

in src/zpotrf_mgpu_right.cpp
### Prototype
```c
void magma_zherk_mgpu2( magma_int_t ngpu, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset, double beta, magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset, magma_int_t nqueue, magma_queue_t queues[][10] );
```
"""
function magma_zherk_mgpu2(ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)
    ccall((:magma_zherk_mgpu2, libmagma), Cvoid, (magma_int_t, magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{NTuple{10, magma_queue_t}}), ngpu, uplo, trans, nb, n, k, alpha, dB, lddb, b_offset, beta, dC, lddc, c_offset, nqueue, queues)
end

"""
    magmablas_zdiinertia(n, dA, ldda, dneig, queue)

LAPACK auxiliary functions (alphabetical order)
### Prototype
```c
magma_int_t magmablas_zdiinertia( magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, int *dneig, magma_queue_t queue );
```
"""
function magmablas_zdiinertia(n, dA, ldda, dneig, queue)
    ccall((:magmablas_zdiinertia, libmagma), magma_int_t, (magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{Cint}, magma_queue_t), n, dA, ldda, dneig, queue)
end

"""
    magmablas_zgeam(transA, transB, m, n, alpha, dA, ldda, beta, dB, lddb, dC, lddc, queue)


### Prototype
```c
void magmablas_zgeam( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex beta, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dC, magma_int_t lddc, magma_queue_t queue );
```
"""
function magmablas_zgeam(transA, transB, m, n, alpha, dA, ldda, beta, dB, lddb, dC, lddc, queue)
    ccall((:magmablas_zgeam, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), transA, transB, m, n, alpha, dA, ldda, beta, dB, lddb, dC, lddc, queue)
end

"""
    magmablas_zheinertia(uplo, n, dA, ldda, ipiv, dneig, queue)


### Prototype
```c
magma_int_t magmablas_zheinertia( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magma_int_t *ipiv, int *dneig, magma_queue_t queue );
```
"""
function magmablas_zheinertia(uplo, n, dA, ldda, ipiv, dneig, queue)
    ccall((:magmablas_zheinertia, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{Cint}, magma_queue_t), uplo, n, dA, ldda, ipiv, dneig, queue)
end

"""
    magma_zlaswp_rowparallel_native(n, input, ldi, output, ldo, k1, k2, pivinfo, queue)


### Prototype
```c
void magma_zlaswp_rowparallel_native( magma_int_t n, magmaDoubleComplex* input, magma_int_t ldi, magmaDoubleComplex* output, magma_int_t ldo, magma_int_t k1, magma_int_t k2, magma_int_t *pivinfo, magma_queue_t queue);
```
"""
function magma_zlaswp_rowparallel_native(n, input, ldi, output, ldo, k1, k2, pivinfo, queue)
    ccall((:magma_zlaswp_rowparallel_native, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_queue_t), n, input, ldi, output, ldo, k1, k2, pivinfo, queue)
end

"""
    magma_zlaswp_columnserial(n, dA, lda, k1, k2, dipiv, queue)


### Prototype
```c
void magma_zlaswp_columnserial( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t lda, magma_int_t k1, magma_int_t k2, magma_int_t *dipiv, magma_queue_t queue);
```
"""
function magma_zlaswp_columnserial(n, dA, lda, k1, k2, dipiv, queue)
    ccall((:magma_zlaswp_columnserial, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_queue_t), n, dA, lda, k1, k2, dipiv, queue)
end

"""
    magmablas_ztrsv(uplo, transA, diag, n, dA, ldda, db, incb, queue)

Level 2 BLAS (alphabetical order)
/
trsv were always queue versions
### Prototype
```c
void magmablas_ztrsv( magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr db, magma_int_t incb, magma_queue_t queue );
```
"""
function magmablas_ztrsv(uplo, transA, diag, n, dA, ldda, db, incb, queue)
    ccall((:magmablas_ztrsv, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, transA, diag, n, dA, ldda, db, incb, queue)
end

"""
    magmablas_ztrsv_outofplace(uplo, transA, diag, n, dA, ldda, db, incb, dx, queue, flag)

todo: move flag before queue?
### Prototype
```c
void magmablas_ztrsv_outofplace( magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr db, magma_int_t incb, magmaDoubleComplex_ptr dx, magma_queue_t queue, magma_int_t flag );
```
"""
function magmablas_ztrsv_outofplace(uplo, transA, diag, n, dA, ldda, db, incb, dx, queue, flag)
    ccall((:magmablas_ztrsv_outofplace, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_queue_t, magma_int_t), uplo, transA, diag, n, dA, ldda, db, incb, dx, queue, flag)
end

"""
    magmablas_zhemv_work(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, dwork, lwork, queue)

hemv/symv_work were always queue versions
### Prototype
```c
magma_int_t magmablas_zhemv_work( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy, magmaDoubleComplex_ptr dwork, magma_int_t lwork, magma_queue_t queue );
```
"""
function magmablas_zhemv_work(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, dwork, lwork, queue)
    ccall((:magmablas_zhemv_work, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, dwork, lwork, queue)
end

"""
    magmablas_zsymv_work(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, dwork, lwork, queue)


### Prototype
```c
magma_int_t magmablas_zsymv_work( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy, magmaDoubleComplex_ptr dwork, magma_int_t lwork, magma_queue_t queue );
```
"""
function magmablas_zsymv_work(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, dwork, lwork, queue)
    ccall((:magmablas_zsymv_work, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, dwork, lwork, queue)
end

"""
    magma_izamax_native(length, x, incx, ipiv, info, step, gbstep, queue)


### Prototype
```c
magma_int_t magma_izamax_native( magma_int_t length, magmaDoubleComplex_ptr x, magma_int_t incx, magma_int_t* ipiv, magma_int_t *info, magma_int_t step, magma_int_t gbstep, magma_queue_t queue);
```
"""
function magma_izamax_native(length, x, incx, ipiv, info, step, gbstep, queue)
    ccall((:magma_izamax_native, libmagma), magma_int_t, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), length, x, incx, ipiv, info, step, gbstep, queue)
end

"""
    magma_zrotg(a, b, c, s, queue)


### Prototype
```c
void magma_zrotg( magmaDoubleComplex_ptr a, magmaDoubleComplex_ptr b, magmaDouble_ptr c, magmaDoubleComplex_ptr s, magma_queue_t queue );
```
"""
function magma_zrotg(a, b, c, s, queue)
    ccall((:magma_zrotg, libmagma), Cvoid, (magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, magmaDoubleComplex_ptr, magma_queue_t), a, b, c, s, queue)
end

"""
    magma_zscal_zgeru_native(m, n, dA, lda, info, step, gbstep, queue)


### Prototype
```c
magma_int_t magma_zscal_zgeru_native( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t lda, magma_int_t *info, magma_int_t step, magma_int_t gbstep, magma_queue_t queue);
```
"""
function magma_zscal_zgeru_native(m, n, dA, lda, info, step, gbstep, queue)
    ccall((:magma_zscal_zgeru_native, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA, lda, info, step, gbstep, queue)
end

"""
    magma_zswap_native(n, x, incx, step, ipiv, queue)


### Prototype
```c
void magma_zswap_native( magma_int_t n, magmaDoubleComplex_ptr x, magma_int_t incx, magma_int_t step, magma_int_t* ipiv, magma_queue_t queue);
```
"""
function magma_zswap_native(n, x, incx, step, ipiv, queue)
    ccall((:magma_zswap_native, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_queue_t), n, x, incx, step, ipiv, queue)
end

"""
    magma_zsymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)


### Prototype
```c
void magma_zsymv( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy, magma_int_t incy, magma_queue_t queue );
```
"""
function magma_zsymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)
    ccall((:magma_zsymv, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, queue)
end

"""
    magma_zsyr(uplo, n, alpha, dx, incx, dA, ldda, queue)


### Prototype
```c
void magma_zsyr( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_queue_t queue );
```
"""
function magma_zsyr(uplo, n, alpha, dx, incx, dA, ldda, queue)
    ccall((:magma_zsyr, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, alpha, dx, incx, dA, ldda, queue)
end

"""
    magma_zsyr2(uplo, n, alpha, dx, incx, dy, incy, dA, ldda, queue)


### Prototype
```c
void magma_zsyr2( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr dx, magma_int_t incx, magmaDoubleComplex_const_ptr dy, magma_int_t incy, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_queue_t queue );
```
"""
function magma_zsyr2(uplo, n, alpha, dx, incx, dy, incy, dA, ldda, queue)
    ccall((:magma_zsyr2, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, alpha, dx, incx, dy, incy, dA, ldda, queue)
end

"""
    magmablas_ztrmv(uplo, trans, diag, n, dA, ldda, dx, incx, queue)


### Prototype
```c
void magmablas_ztrmv( magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda, magmaDoubleComplex *dx, magma_int_t incx, magma_queue_t queue );
```
"""
function magmablas_ztrmv(uplo, trans, diag, n, dA, ldda, dx, incx, queue)
    ccall((:magmablas_ztrmv, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), uplo, trans, diag, n, dA, ldda, dx, incx, queue)
end

"""
    magmablas_zherk_internal(uplo, trans, n, k, nb, alpha, dA, ldda, dB, lddb, beta, dC, lddc, conjugate, queue)


### Prototype
```c
void magmablas_zherk_internal( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magma_int_t nb, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex_ptr dC, magma_int_t lddc, magma_int_t conjugate, magma_queue_t queue);
```
"""
function magmablas_zherk_internal(uplo, trans, n, k, nb, alpha, dA, ldda, dB, lddb, beta, dC, lddc, conjugate, queue)
    ccall((:magmablas_zherk_internal, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex, magmaDoubleComplex_ptr, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, nb, alpha, dA, ldda, dB, lddb, beta, dC, lddc, conjugate, queue)
end

"""
    magmablas_zherk_small_reduce(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc, nthread_blocks, queue)


### Prototype
```c
void magmablas_zherk_small_reduce( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex* dA, magma_int_t ldda, double beta, magmaDoubleComplex* dC, magma_int_t lddc, magma_int_t nthread_blocks, magma_queue_t queue );
```
"""
function magmablas_zherk_small_reduce(uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc, nthread_blocks, queue)
    ccall((:magmablas_zherk_small_reduce, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Cdouble, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA, ldda, beta, dC, lddc, nthread_blocks, queue)
end

"""
    magmablas_ztrmm(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, queue)


### Prototype
```c
void magmablas_ztrmm( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex *dA, magma_int_t ldda, magmaDoubleComplex *dB, magma_int_t lddb, magma_queue_t queue );
```
"""
function magmablas_ztrmm(side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, queue)
    ccall((:magmablas_ztrmm, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb, queue)
end

"""
    magma_zgetf2trsm_2d_native(m, n, dA, ldda, dB, lddb, queue)


### Prototype
```c
void magma_zgetf2trsm_2d_native( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_queue_t queue);
```
"""
function magma_zgetf2trsm_2d_native(m, n, dA, ldda, dB, lddb, queue)
    ccall((:magma_zgetf2trsm_2d_native, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), m, n, dA, ldda, dB, lddb, queue)
end

"""
    magma_zpotf2_lpout(uplo, n, dA, lda, gbstep, dinfo, queue)


### Prototype
```c
magma_int_t magma_zpotf2_lpout( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *dA, magma_int_t lda, magma_int_t gbstep, magma_int_t *dinfo, magma_queue_t queue);
```
"""
function magma_zpotf2_lpout(uplo, n, dA, lda, gbstep, dinfo, queue)
    ccall((:magma_zpotf2_lpout, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_queue_t), uplo, n, dA, lda, gbstep, dinfo, queue)
end

"""
    magma_zpotf2_lpin(uplo, n, dA, lda, gbstep, dinfo, queue)


### Prototype
```c
magma_int_t magma_zpotf2_lpin( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *dA, magma_int_t lda, magma_int_t gbstep, magma_int_t *dinfo, magma_queue_t queue);
```
"""
function magma_zpotf2_lpin(uplo, n, dA, lda, gbstep, dinfo, queue)
    ccall((:magma_zpotf2_lpin, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_queue_t), uplo, n, dA, lda, gbstep, dinfo, queue)
end

"""
    magma_zset_pointer(output_array, input, lda, row, column, batch_offset, batchCount, queue)

 local auxiliary routines
### Prototype
```c
void magma_zset_pointer( magmaDoubleComplex **output_array, magmaDoubleComplex *input, magma_int_t lda, magma_int_t row, magma_int_t column, magma_int_t batch_offset, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zset_pointer(output_array, input, lda, row, column, batch_offset, batchCount, queue)
    ccall((:magma_zset_pointer, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input, lda, row, column, batch_offset, batchCount, queue)
end

"""
    magma_zdisplace_pointers(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_zdisplace_pointers( magmaDoubleComplex **output_array, magmaDoubleComplex **input_array, magma_int_t lda, magma_int_t row, magma_int_t column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zdisplace_pointers(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_zdisplace_pointers, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    magma_zrecommend_cublas_gemm_batched(transa, transb, m, n, k)


### Prototype
```c
magma_int_t magma_zrecommend_cublas_gemm_batched( magma_trans_t transa, magma_trans_t transb, magma_int_t m, magma_int_t n, magma_int_t k);
```
"""
function magma_zrecommend_cublas_gemm_batched(transa, transb, m, n, k)
    ccall((:magma_zrecommend_cublas_gemm_batched, libmagma), magma_int_t, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t), transa, transb, m, n, k)
end

"""
    magma_zrecommend_cublas_gemm_stream(transa, transb, m, n, k)


### Prototype
```c
magma_int_t magma_zrecommend_cublas_gemm_stream( magma_trans_t transa, magma_trans_t transb, magma_int_t m, magma_int_t n, magma_int_t k);
```
"""
function magma_zrecommend_cublas_gemm_stream(transa, transb, m, n, k)
    ccall((:magma_zrecommend_cublas_gemm_stream, libmagma), magma_int_t, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t), transa, transb, m, n, k)
end

"""
    magma_get_zpotrf_batched_nbparam(n, nb, recnb)


### Prototype
```c
void magma_get_zpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);
```
"""
function magma_get_zpotrf_batched_nbparam(n, nb, recnb)
    ccall((:magma_get_zpotrf_batched_nbparam, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, recnb)
end

# no prototype is found for this function at magma_zbatched.h:55:13, please use with caution
"""
    magma_get_zpotrf_batched_crossover()


### Prototype
```c
magma_int_t magma_get_zpotrf_batched_crossover();
```
"""
function magma_get_zpotrf_batched_crossover()
    ccall((:magma_get_zpotrf_batched_crossover, libmagma), magma_int_t, ())
end

"""
    magma_get_zgetrf_batched_nbparam(n, nb, recnb)


### Prototype
```c
void magma_get_zgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);
```
"""
function magma_get_zgetrf_batched_nbparam(n, nb, recnb)
    ccall((:magma_get_zgetrf_batched_nbparam, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, recnb)
end

"""
    magma_get_zgetrf_batched_ntcol(m, n)


### Prototype
```c
magma_int_t magma_get_zgetrf_batched_ntcol(magma_int_t m, magma_int_t n);
```
"""
function magma_get_zgetrf_batched_ntcol(m, n)
    ccall((:magma_get_zgetrf_batched_ntcol, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgemm_batched_ntcol(n)


### Prototype
```c
magma_int_t magma_get_zgemm_batched_ntcol(magma_int_t n);
```
"""
function magma_get_zgemm_batched_ntcol(n)
    ccall((:magma_get_zgemm_batched_ntcol, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zgemm_batched_smallsq_limit(n)


### Prototype
```c
magma_int_t magma_get_zgemm_batched_smallsq_limit(magma_int_t n);
```
"""
function magma_get_zgemm_batched_smallsq_limit(n)
    ccall((:magma_get_zgemm_batched_smallsq_limit, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zgeqrf_batched_nb(m)


### Prototype
```c
magma_int_t magma_get_zgeqrf_batched_nb(magma_int_t m);
```
"""
function magma_get_zgeqrf_batched_nb(m)
    ccall((:magma_get_zgeqrf_batched_nb, libmagma), magma_int_t, (magma_int_t,), m)
end

"""
    magma_use_zgeqrf_batched_fused_update(m, n, batchCount)


### Prototype
```c
magma_int_t magma_use_zgeqrf_batched_fused_update(magma_int_t m, magma_int_t n, magma_int_t batchCount);
```
"""
function magma_use_zgeqrf_batched_fused_update(m, n, batchCount)
    ccall((:magma_use_zgeqrf_batched_fused_update, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t), m, n, batchCount)
end

"""
    magma_get_zgeqr2_fused_sm_batched_nthreads(m, n)


### Prototype
```c
magma_int_t magma_get_zgeqr2_fused_sm_batched_nthreads(magma_int_t m, magma_int_t n);
```
"""
function magma_get_zgeqr2_fused_sm_batched_nthreads(m, n)
    ccall((:magma_get_zgeqr2_fused_sm_batched_nthreads, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgeqrf_batched_ntcol(m, n)


### Prototype
```c
magma_int_t magma_get_zgeqrf_batched_ntcol(magma_int_t m, magma_int_t n);
```
"""
function magma_get_zgeqrf_batched_ntcol(m, n)
    ccall((:magma_get_zgeqrf_batched_ntcol, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgetri_batched_ntcol(m, n)


### Prototype
```c
magma_int_t magma_get_zgetri_batched_ntcol(magma_int_t m, magma_int_t n);
```
"""
function magma_get_zgetri_batched_ntcol(m, n)
    ccall((:magma_get_zgetri_batched_ntcol, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_ztrsm_batched_stop_nb(side, m, n)


### Prototype
```c
magma_int_t magma_get_ztrsm_batched_stop_nb(magma_side_t side, magma_int_t m, magma_int_t n);
```
"""
function magma_get_ztrsm_batched_stop_nb(side, m, n)
    ccall((:magma_get_ztrsm_batched_stop_nb, libmagma), magma_int_t, (magma_side_t, magma_int_t, magma_int_t), side, m, n)
end

"""
    magmablas_zswapdblk_batched(n, nb, dA, ldda, inca, dB, lddb, incb, batchCount, queue)


### Prototype
```c
void magmablas_zswapdblk_batched( magma_int_t n, magma_int_t nb, magmaDoubleComplex **dA, magma_int_t ldda, magma_int_t inca, magmaDoubleComplex **dB, magma_int_t lddb, magma_int_t incb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zswapdblk_batched(n, nb, dA, ldda, inca, dB, lddb, incb, batchCount, queue)
    ccall((:magmablas_zswapdblk_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), n, nb, dA, ldda, inca, dB, lddb, incb, batchCount, queue)
end

"""
    magmablas_zgemm_batched_core(transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)

 BLAS batched routines
### Prototype
```c
void magmablas_zgemm_batched_core( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_batched_core(transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)
    ccall((:magmablas_zgemm_batched_core, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)
end

"""
    magma_zgemm_batched_core(transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)


### Prototype
```c
void magma_zgemm_batched_core( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgemm_batched_core(transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)
    ccall((:magma_zgemm_batched_core, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)
end

"""
    magma_zgemm_batched(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magma_zgemm_batched( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgemm_batched(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magma_zgemm_batched, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zgemm_batched(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_batched( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_batched(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zgemm_batched, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zgemm_batched_strided(transA, transB, m, n, k, alpha, dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, strideC, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_batched_strided( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * dA, magma_int_t ldda, magma_int_t strideA, magmaDoubleComplex const * dB, magma_int_t lddb, magma_int_t strideB, magmaDoubleComplex beta, magmaDoubleComplex * dC, magma_int_t lddc, magma_int_t strideC, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_batched_strided(transA, transB, m, n, k, alpha, dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, strideC, batchCount, queue)
    ccall((:magmablas_zgemm_batched_strided, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA, ldda, strideA, dB, lddb, strideB, beta, dC, lddc, strideC, batchCount, queue)
end

"""
    magmablas_zgemm_batched_smallsq(transA, transB, m, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_batched_smallsq( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_batched_smallsq(transA, transB, m, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)
    ccall((:magmablas_zgemm_batched_smallsq, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)
end

"""
    magmablas_zsyrk_batched_core(uplo, trans, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyrk_batched_core( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyrk_batched_core(uplo, trans, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)
    ccall((:magmablas_zsyrk_batched_core, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)
end

"""
    magmablas_zherk_batched_core(uplo, trans, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zherk_batched_core( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zherk_batched_core(uplo, trans, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)
    ccall((:magmablas_zherk_batched_core, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta, dC_array, ci, cj, lddc, batchCount, queue)
end

"""
    magmablas_zsyrk_batched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyrk_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyrk_batched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zsyrk_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magma_zherk_batched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magma_zherk_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, double beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zherk_batched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magma_zherk_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zherk_batched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zherk_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, double beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zherk_batched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zherk_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zher2k_batched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zher2k_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t lddb, double beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zher2k_batched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zher2k_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zsyr2k_batched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyr2k_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t ldda, magmaDoubleComplex const * const * dB_array, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyr2k_batched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zsyr2k_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_ztrtri_diag_batched(uplo, diag, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue)


### Prototype
```c
void magmablas_ztrtri_diag_batched( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, magmaDoubleComplex const * const *dA_array, magma_int_t ldda, magmaDoubleComplex **dinvA_array, magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrtri_diag_batched(uplo, diag, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue)
    ccall((:magmablas_ztrtri_diag_batched, libmagma), Cvoid, (magma_uplo_t, magma_diag_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, diag, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue)
end

"""
    magmablas_ztrsm_small_batched(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_small_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrsm_small_batched(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_small_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_recursive_batched(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_recursive_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrsm_recursive_batched(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_recursive_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_batched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrsm_batched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_inv_batched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_inv_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** dB_array, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_batched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_inv_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_inv_work_batched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_inv_work_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t flag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** dB_array, magma_int_t lddb, magmaDoubleComplex** dX_array, magma_int_t lddx, magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length, magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ, magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_work_batched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, queue)
    ccall((:magmablas_ztrsm_inv_work_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, queue)
end

"""
    magmablas_ztrsm_inv_outofplace_batched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_inv_outofplace_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t flag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** dB_array, magma_int_t lddb, magmaDoubleComplex** dX_array, magma_int_t lddx, magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length, magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ, magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_outofplace_batched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, queue)
    ccall((:magmablas_ztrsm_inv_outofplace_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, queue)
end

"""
    magmablas_ztrsv_batched(uplo, transA, diag, n, dA_array, ldda, dB_array, incb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsv_batched( magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** dB_array, magma_int_t incb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsv_batched(uplo, transA, diag, n, dA_array, ldda, dB_array, incb, batchCount, queue)
    ccall((:magmablas_ztrsv_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, transA, diag, n, dA_array, ldda, dB_array, incb, batchCount, queue)
end

"""
    magmablas_ztrsv_work_batched(uplo, transA, diag, n, dA_array, ldda, dB_array, incb, dX_array, batchCount, queue)


### Prototype
```c
void magmablas_ztrsv_work_batched( magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** dB_array, magma_int_t incb, magmaDoubleComplex** dX_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsv_work_batched(uplo, transA, diag, n, dA_array, ldda, dB_array, incb, dX_array, batchCount, queue)
    ccall((:magmablas_ztrsv_work_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_queue_t), uplo, transA, diag, n, dA_array, ldda, dB_array, incb, dX_array, batchCount, queue)
end

"""
    magmablas_ztrsv_outofplace_batched(uplo, transA, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue, flag)


### Prototype
```c
void magmablas_ztrsv_outofplace_batched( magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t n, magmaDoubleComplex ** A_array, magma_int_t lda, magmaDoubleComplex **b_array, magma_int_t incb, magmaDoubleComplex **x_array, magma_int_t batchCount, magma_queue_t queue, magma_int_t flag);
```
"""
function magmablas_ztrsv_outofplace_batched(uplo, transA, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue, flag)
    ccall((:magmablas_ztrsv_outofplace_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_queue_t, magma_int_t), uplo, transA, diag, n, A_array, lda, b_array, incb, x_array, batchCount, queue, flag)
end

"""
    magmablas_ztrmm_batched_core(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_batched_core( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_batched_core(side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_batched_core, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_ztrmm_batched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_batched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_zhemm_batched_core(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, roffA, coffA, roffB, coffB, roffC, coffC, batchCount, queue)


### Prototype
```c
void magmablas_zhemm_batched_core( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemm_batched_core(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, roffA, coffA, roffB, coffB, roffC, coffC, batchCount, queue)
    ccall((:magmablas_zhemm_batched_core, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, roffA, coffA, roffB, coffB, roffC, coffC, batchCount, queue)
end

"""
    magmablas_zhemm_batched(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zhemm_batched( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemm_batched(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zhemm_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zhemv_batched_core(uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, offA, offX, offY, batchCount, queue)


### Prototype
```c
void magmablas_zhemv_batched_core( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dX_array, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex **dY_array, magma_int_t incy, magma_int_t offA, magma_int_t offX, magma_int_t offY, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemv_batched_core(uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, offA, offX, offY, batchCount, queue)
    ccall((:magmablas_zhemv_batched_core, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, offA, offX, offY, batchCount, queue)
end

"""
    magmablas_zhemv_batched(uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zhemv_batched( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dX_array, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex **dY_array, magma_int_t incy, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemv_batched(uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, batchCount, queue)
    ccall((:magmablas_zhemv_batched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, batchCount, queue)
end

"""
    magma_zpotrf_batched(uplo, n, dA_array, lda, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_batched( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_batched(uplo, n, dA_array, lda, info_array, batchCount, queue)
    ccall((:magma_zpotrf_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, dA_array, lda, info_array, batchCount, queue)
end

"""
    magma_zpotf2_batched(uplo, n, dA_array, ai, aj, lda, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotf2_batched( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotf2_batched(uplo, n, dA_array, ai, aj, lda, info_array, gbstep, batchCount, queue)
    ccall((:magma_zpotf2_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, dA_array, ai, aj, lda, info_array, gbstep, batchCount, queue)
end

"""
    magma_zpotrf_panel_batched(uplo, n, nb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_panel_batched( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_panel_batched(uplo, n, nb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
    ccall((:magma_zpotrf_panel_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, nb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
end

"""
    magma_zpotrf_recpanel_batched(uplo, m, n, min_recpnb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_recpanel_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb, magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_recpanel_batched(uplo, m, n, min_recpnb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
    ccall((:magma_zpotrf_recpanel_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, m, n, min_recpnb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
end

"""
    magma_zpotrf_rectile_batched(uplo, m, n, min_recpnb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_rectile_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t min_recpnb, magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_rectile_batched(uplo, m, n, min_recpnb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
    ccall((:magma_zpotrf_rectile_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, m, n, min_recpnb, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
end

"""
    magma_zpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrs_batched( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magma_zpotrs_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magma_zposv_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, dinfo_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zposv_batched( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t *dinfo_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zposv_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, dinfo_array, batchCount, queue)
    ccall((:magma_zposv_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, nrhs, dA_array, ldda, dB_array, lddb, dinfo_array, batchCount, queue)
end

"""
    magma_zgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrs_batched( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t **dipiv_array, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchCount, queue)
    ccall((:magma_zgetrs_batched, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchCount, queue)
end

"""
    magma_zlaswp_rowparallel_batched(n, input_array, input_i, input_j, ldi, output_array, output_i, output_j, ldo, k1, k2, pivinfo_array, batchCount, queue)


### Prototype
```c
void magma_zlaswp_rowparallel_batched( magma_int_t n, magmaDoubleComplex** input_array, magma_int_t input_i, magma_int_t input_j, magma_int_t ldi, magmaDoubleComplex** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo, magma_int_t k1, magma_int_t k2, magma_int_t **pivinfo_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_rowparallel_batched(n, input_array, input_i, input_j, ldi, output_array, output_i, output_j, ldo, k1, k2, pivinfo_array, batchCount, queue)
    ccall((:magma_zlaswp_rowparallel_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_queue_t), n, input_array, input_i, input_j, ldi, output_array, output_i, output_j, ldo, k1, k2, pivinfo_array, batchCount, queue)
end

"""
    magma_zlaswp_rowserial_batched(n, dA_array, lda, k1, k2, ipiv_array, batchCount, queue)


### Prototype
```c
void magma_zlaswp_rowserial_batched( magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda, magma_int_t k1, magma_int_t k2, magma_int_t **ipiv_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_rowserial_batched(n, dA_array, lda, k1, k2, ipiv_array, batchCount, queue)
    ccall((:magma_zlaswp_rowserial_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_queue_t), n, dA_array, lda, k1, k2, ipiv_array, batchCount, queue)
end

"""
    magma_zlaswp_columnserial_batched(n, dA_array, lda, k1, k2, ipiv_array, batchCount, queue)


### Prototype
```c
void magma_zlaswp_columnserial_batched( magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda, magma_int_t k1, magma_int_t k2, magma_int_t **ipiv_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_columnserial_batched(n, dA_array, lda, k1, k2, ipiv_array, batchCount, queue)
    ccall((:magma_zlaswp_columnserial_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_queue_t), n, dA_array, lda, k1, k2, ipiv_array, batchCount, queue)
end

"""
    magmablas_ztranspose_batched(m, n, dA_array, ldda, dAT_array, lddat, batchCount, queue)


### Prototype
```c
void magmablas_ztranspose_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dAT_array, magma_int_t lddat, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztranspose_batched(m, n, dA_array, ldda, dAT_array, lddat, batchCount, queue)
    ccall((:magmablas_ztranspose_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, ldda, dAT_array, lddat, batchCount, queue)
end

"""
    magmablas_zlaset_internal_batched(uplo, m, n, offdiag, diag, dAarray, Ai, Aj, ldda, batchCount, queue)


### Prototype
```c
void magmablas_zlaset_internal_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex offdiag, magmaDoubleComplex diag, magmaDoubleComplex_ptr dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zlaset_internal_batched(uplo, m, n, offdiag, diag, dAarray, Ai, Aj, ldda, batchCount, queue)
    ccall((:magmablas_zlaset_internal_batched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, m, n, offdiag, diag, dAarray, Ai, Aj, ldda, batchCount, queue)
end

"""
    magmablas_zlaset_batched(uplo, m, n, offdiag, diag, dAarray, ldda, batchCount, queue)


### Prototype
```c
void magmablas_zlaset_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex offdiag, magmaDoubleComplex diag, magmaDoubleComplex_ptr dAarray[], magma_int_t ldda, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zlaset_batched(uplo, m, n, offdiag, diag, dAarray, ldda, batchCount, queue)
    ccall((:magmablas_zlaset_batched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), uplo, m, n, offdiag, diag, dAarray, ldda, batchCount, queue)
end

"""
    magma_zgesv_batched_small(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgesv_batched_small( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t* dinfo_array, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgesv_batched_small(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue)
    ccall((:magma_zgesv_batched_small, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue)
end

"""
    magma_zgetf2_batched(m, n, dA_array, ai, aj, lda, ipiv_array, dpivinfo_array, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetf2_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t **ipiv_array, magma_int_t **dpivinfo_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetf2_batched(m, n, dA_array, ai, aj, lda, ipiv_array, dpivinfo_array, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetf2_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, ai, aj, lda, ipiv_array, dpivinfo_array, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetrf_recpanel_batched(m, n, min_recpnb, dA_array, ai, aj, ldda, dipiv_array, dpivinfo_array, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_recpanel_batched( magma_int_t m, magma_int_t n, magma_int_t min_recpnb, magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t** dipiv_array, magma_int_t** dpivinfo_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_recpanel_batched(m, n, min_recpnb, dA_array, ai, aj, ldda, dipiv_array, dpivinfo_array, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetrf_recpanel_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, min_recpnb, dA_array, ai, aj, ldda, dipiv_array, dpivinfo_array, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetrf_batched(m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, magma_int_t **ipiv_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_batched(m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue)
    ccall((:magma_zgetrf_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, dA_array, lda, ipiv_array, info_array, batchCount, queue)
end

"""
    magma_zgetf2_fused_batched(m, n, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetf2_fused_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t **dipiv_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetf2_fused_batched(m, n, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue)
    ccall((:magma_zgetf2_fused_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, dA_array, ai, aj, ldda, dipiv_array, info_array, batchCount, queue)
end

"""
    magma_zgetrf_batched_smallsq_noshfl(n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_batched_smallsq_noshfl( magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgetrf_batched_smallsq_noshfl(n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)
    ccall((:magma_zgetrf_batched_smallsq_noshfl, libmagma), magma_int_t, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)
end

"""
    magma_zgetrf_batched_smallsq_shfl(n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_batched_smallsq_shfl( magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgetrf_batched_smallsq_shfl(n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)
    ccall((:magma_zgetrf_batched_smallsq_shfl, libmagma), magma_int_t, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)
end

"""
    magma_zgetri_outofplace_batched(n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetri_outofplace_batched( magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t **dipiv_array, magmaDoubleComplex **dinvA_array, magma_int_t lddia, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetri_outofplace_batched(n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue)
    ccall((:magma_zgetri_outofplace_batched, libmagma), magma_int_t, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, dA_array, ldda, dipiv_array, dinvA_array, lddia, info_array, batchCount, queue)
end

"""
    magma_zdisplace_intpointers(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_zdisplace_intpointers( magma_int_t **output_array, magma_int_t **input_array, magma_int_t lda, magma_int_t row, magma_int_t column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zdisplace_intpointers(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_zdisplace_intpointers, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    magmablas_izamax_atomic_batched(n, x_array, incx, max_id_array, batchCount)


### Prototype
```c
void magmablas_izamax_atomic_batched( magma_int_t n, magmaDoubleComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);
```
"""
function magmablas_izamax_atomic_batched(n, x_array, incx, max_id_array, batchCount)
    ccall((:magmablas_izamax_atomic_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t), n, x_array, incx, max_id_array, batchCount)
end

"""
    magmablas_izamax_tree_batched(n, x_array, incx, max_id_array, batchCount)


### Prototype
```c
void magmablas_izamax_tree_batched( magma_int_t n, magmaDoubleComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);
```
"""
function magmablas_izamax_tree_batched(n, x_array, incx, max_id_array, batchCount)
    ccall((:magmablas_izamax_tree_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t), n, x_array, incx, max_id_array, batchCount)
end

"""
    magmablas_izamax_batched(n, x_array, incx, max_id_array, batchCount)


### Prototype
```c
void magmablas_izamax_batched( magma_int_t n, magmaDoubleComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);
```
"""
function magmablas_izamax_batched(n, x_array, incx, max_id_array, batchCount)
    ccall((:magmablas_izamax_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t), n, x_array, incx, max_id_array, batchCount)
end

"""
    magmablas_izamax(n, x, incx, max_id)


### Prototype
```c
void magmablas_izamax( magma_int_t n, magmaDoubleComplex* x, magma_int_t incx, magma_int_t *max_id);
```
"""
function magmablas_izamax(n, x, incx, max_id)
    ccall((:magmablas_izamax, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), n, x, incx, max_id)
end

"""
    magma_izamax_batched(length, x_array, xi, xj, lda, incx, ipiv_array, ipiv_i, step, gbstep, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_izamax_batched( magma_int_t length, magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t lda, magma_int_t incx, magma_int_t** ipiv_array, magma_int_t ipiv_i, magma_int_t step, magma_int_t gbstep, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_izamax_batched(length, x_array, xi, xj, lda, incx, ipiv_array, ipiv_i, step, gbstep, info_array, batchCount, queue)
    ccall((:magma_izamax_batched, libmagma), magma_int_t, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), length, x_array, xi, xj, lda, incx, ipiv_array, ipiv_i, step, gbstep, info_array, batchCount, queue)
end

"""
    magma_zswap_batched(n, x_array, xi, xj, incx, step, ipiv_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zswap_batched( magma_int_t n, magmaDoubleComplex **x_array, magma_int_t xi, magma_int_t xj, magma_int_t incx, magma_int_t step, magma_int_t** ipiv_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zswap_batched(n, x_array, xi, xj, incx, step, ipiv_array, batchCount, queue)
    ccall((:magma_zswap_batched, libmagma), magma_int_t, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_queue_t), n, x_array, xi, xj, incx, step, ipiv_array, batchCount, queue)
end

"""
    magma_zscal_zgeru_batched(m, n, dA_array, ai, aj, lda, info_array, step, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zscal_zgeru_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t *info_array, magma_int_t step, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zscal_zgeru_batched(m, n, dA_array, ai, aj, lda, info_array, step, gbstep, batchCount, queue)
    ccall((:magma_zscal_zgeru_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, ai, aj, lda, info_array, step, gbstep, batchCount, queue)
end

"""
    magma_zcomputecolumn_batched(m, paneloffset, step, dA_array, lda, ai, aj, ipiv_array, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zcomputecolumn_batched( magma_int_t m, magma_int_t paneloffset, magma_int_t step, magmaDoubleComplex **dA_array, magma_int_t lda, magma_int_t ai, magma_int_t aj, magma_int_t **ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zcomputecolumn_batched(m, paneloffset, step, dA_array, lda, ai, aj, ipiv_array, info_array, gbstep, batchCount, queue)
    ccall((:magma_zcomputecolumn_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, paneloffset, step, dA_array, lda, ai, aj, ipiv_array, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetf2trsm_batched(ib, n, dA_array, j, lda, batchCount, queue)


### Prototype
```c
void magma_zgetf2trsm_batched( magma_int_t ib, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t j, magma_int_t lda, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetf2trsm_batched(ib, n, dA_array, j, lda, batchCount, queue)
    ccall((:magma_zgetf2trsm_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), ib, n, dA_array, j, lda, batchCount, queue)
end

"""
    magma_zgetf2_nopiv_internal_batched(m, n, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetf2_nopiv_internal_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t* info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgetf2_nopiv_internal_batched(m, n, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetf2_nopiv_internal_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetf2_nopiv_batched(m, n, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetf2_nopiv_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetf2_nopiv_batched(m, n, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetf2_nopiv_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, ai, aj, ldda, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetrf_recpanel_nopiv_batched(m, n, min_recpnb, dA_array, ldda, dX_array, dX_length, dinvA_array, dinvA_length, dW1_displ, dW2_displ, dW3_displ, dW4_displ, dW5_displ, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_recpanel_nopiv_batched( magma_int_t m, magma_int_t n, magma_int_t min_recpnb, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** dX_array, magma_int_t dX_length, magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length, magmaDoubleComplex** dW1_displ, magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ, magmaDoubleComplex** dW4_displ, magmaDoubleComplex** dW5_displ, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_recpanel_nopiv_batched(m, n, min_recpnb, dA_array, ldda, dX_array, dX_length, dinvA_array, dinvA_length, dW1_displ, dW2_displ, dW3_displ, dW4_displ, dW5_displ, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetrf_recpanel_nopiv_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, min_recpnb, dA_array, ldda, dX_array, dX_length, dinvA_array, dinvA_length, dW1_displ, dW2_displ, dW3_displ, dW4_displ, dW5_displ, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetrf_nopiv_batched(m, n, dA_array, lda, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_nopiv_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_nopiv_batched(m, n, dA_array, lda, info_array, batchCount, queue)
    ccall((:magma_zgetrf_nopiv_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, dA_array, lda, info_array, batchCount, queue)
end

"""
    magma_zgetrs_nopiv_batched(trans, n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrs_nopiv_batched( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrs_nopiv_batched(trans, n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)
    ccall((:magma_zgetrs_nopiv_batched, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), trans, n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)
end

"""
    magma_zgesv_nopiv_batched(n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgesv_nopiv_batched( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgesv_nopiv_batched(n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)
    ccall((:magma_zgesv_nopiv_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)
end

"""
    magma_zgesv_rbt_batched(n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgesv_rbt_batched( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgesv_rbt_batched(n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)
    ccall((:magma_zgesv_rbt_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, nrhs, dA_array, ldda, dB_array, lddb, info_array, batchCount, queue)
end

"""
    magma_zgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgesv_batched( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t **dipiv_array, magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t *dinfo_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue)
    ccall((:magma_zgesv_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batchCount, queue)
end

"""
    magma_zgerbt_batched(gen, n, nrhs, dA_array, ldda, dB_array, lddb, U, V, info, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgerbt_batched( magma_bool_t gen, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magmaDoubleComplex *U, magmaDoubleComplex *V, magma_int_t *info, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgerbt_batched(gen, n, nrhs, dA_array, ldda, dB_array, lddb, U, V, info, batchCount, queue)
    ccall((:magma_zgerbt_batched, libmagma), magma_int_t, (magma_bool_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), gen, n, nrhs, dA_array, ldda, dB_array, lddb, U, V, info, batchCount, queue)
end

"""
    magmablas_zprbt_batched(n, dA_array, ldda, du, dv, batchCount, queue)


### Prototype
```c
void magmablas_zprbt_batched( magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex *du, magmaDoubleComplex *dv, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zprbt_batched(n, dA_array, ldda, du, dv, batchCount, queue)
    ccall((:magmablas_zprbt_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), n, dA_array, ldda, du, dv, batchCount, queue)
end

"""
    magmablas_zprbt_mv_batched(n, dv, db_array, batchCount, queue)


### Prototype
```c
void magmablas_zprbt_mv_batched( magma_int_t n, magmaDoubleComplex *dv, magmaDoubleComplex **db_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zprbt_mv_batched(n, dv, db_array, batchCount, queue)
    ccall((:magmablas_zprbt_mv_batched, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_queue_t), n, dv, db_array, batchCount, queue)
end

"""
    magmablas_zprbt_mtv_batched(n, du, db_array, batchCount, queue)


### Prototype
```c
void magmablas_zprbt_mtv_batched( magma_int_t n, magmaDoubleComplex *du, magmaDoubleComplex **db_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zprbt_mtv_batched(n, du, db_array, batchCount, queue)
    ccall((:magmablas_zprbt_mtv_batched, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_queue_t), n, du, db_array, batchCount, queue)
end

"""
    setup_pivinfo(pivinfo, ipiv, m, nb, queue)


### Prototype
```c
void setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, magma_int_t m, magma_int_t nb, magma_queue_t queue);
```
"""
function setup_pivinfo(pivinfo, ipiv, m, nb, queue)
    ccall((:setup_pivinfo, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), pivinfo, ipiv, m, nb, queue)
end

"""
    magmablas_zgeadd_batched(m, n, alpha, dAarray, ldda, dBarray, lddb, batchCount, queue)


### Prototype
```c
void magmablas_zgeadd_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_const_ptr const dAarray[], magma_int_t ldda, magmaDoubleComplex_ptr dBarray[], magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgeadd_batched(m, n, alpha, dAarray, ldda, dBarray, lddb, batchCount, queue)
    ccall((:magmablas_zgeadd_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), m, n, alpha, dAarray, ldda, dBarray, lddb, batchCount, queue)
end

"""
    magmablas_zlacpy_internal_batched(uplo, m, n, dAarray, Ai, Aj, ldda, dBarray, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_zlacpy_internal_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr const dAarray[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex_ptr dBarray[], magma_int_t Bi, magma_int_t Bj, magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zlacpy_internal_batched(uplo, m, n, dAarray, Ai, Aj, ldda, dBarray, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_zlacpy_internal_batched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, m, n, dAarray, Ai, Aj, ldda, dBarray, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_zlacpy_batched(uplo, m, n, dAarray, ldda, dBarray, lddb, batchCount, queue)


### Prototype
```c
void magmablas_zlacpy_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr const dAarray[], magma_int_t ldda, magmaDoubleComplex_ptr dBarray[], magma_int_t lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zlacpy_batched(uplo, m, n, dAarray, ldda, dBarray, lddb, batchCount, queue)
    ccall((:magmablas_zlacpy_batched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), uplo, m, n, dAarray, ldda, dBarray, lddb, batchCount, queue)
end

"""
    magmablas_zgemv_batched_template(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zgemv_batched_template( magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zgemv_batched_template(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
    ccall((:magmablas_zgemv_batched_template, libmagma), Cvoid, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
end

"""
    magmablas_zgemv_batched(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zgemv_batched( magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zgemv_batched(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
    ccall((:magmablas_zgemv_batched, libmagma), Cvoid, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
end

"""
    magma_zgeqrf_batched_smallsq(n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqrf_batched_smallsq( magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t* info_array, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgeqrf_batched_smallsq(n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue)
    ccall((:magma_zgeqrf_batched_smallsq, libmagma), magma_int_t, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue)
end

"""
    magma_zgeqrf_batched(m, n, dA_array, lda, dtau_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqrf_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, magmaDoubleComplex **dtau_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgeqrf_batched(m, n, dA_array, lda, dtau_array, info_array, batchCount, queue)
    ccall((:magma_zgeqrf_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, dA_array, lda, dtau_array, info_array, batchCount, queue)
end

"""
    magma_zgeqrf_expert_batched(m, n, nb, dA_array, ldda, dR_array, lddr, dT_array, lddt, dtau_array, provide_RT, dW_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqrf_expert_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dR_array, magma_int_t lddr, magmaDoubleComplex **dT_array, magma_int_t lddt, magmaDoubleComplex **dtau_array, magma_int_t provide_RT, magmaDoubleComplex **dW_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgeqrf_expert_batched(m, n, nb, dA_array, ldda, dR_array, lddr, dT_array, lddt, dtau_array, provide_RT, dW_array, info_array, batchCount, queue)
    ccall((:magma_zgeqrf_expert_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, nb, dA_array, ldda, dR_array, lddr, dT_array, lddt, dtau_array, provide_RT, dW_array, info_array, batchCount, queue)
end

"""
    magma_zgeqrf_batched_v4(m, n, dA_array, lda, tau_array, info_array, batchCount)


### Prototype
```c
magma_int_t magma_zgeqrf_batched_v4( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, magmaDoubleComplex **tau_array, magma_int_t *info_array, magma_int_t batchCount);
```
"""
function magma_zgeqrf_batched_v4(m, n, dA_array, lda, tau_array, info_array, batchCount)
    ccall((:magma_zgeqrf_batched_v4, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t), m, n, dA_array, lda, tau_array, info_array, batchCount)
end

"""
    magma_zgeqrf_panel_fused_update_batched(m, n, nb, dA_array, Ai, Aj, ldda, tau_array, taui, dR_array, Ri, Rj, lddr, info_array, separate_R_V, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqrf_panel_fused_update_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex** tau_array, magma_int_t taui, magmaDoubleComplex** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr, magma_int_t *info_array, magma_int_t separate_R_V, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgeqrf_panel_fused_update_batched(m, n, nb, dA_array, Ai, Aj, ldda, tau_array, taui, dR_array, Ri, Rj, lddr, info_array, separate_R_V, batchCount, queue)
    ccall((:magma_zgeqrf_panel_fused_update_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, nb, dA_array, Ai, Aj, ldda, tau_array, taui, dR_array, Ri, Rj, lddr, info_array, separate_R_V, batchCount, queue)
end

"""
    magma_zgeqrf_panel_internal_batched(m, n, nb, dA_array, Ai, Aj, ldda, tau_array, taui, dT_array, Ti, Tj, lddt, dR_array, Ri, Rj, lddr, dwork_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqrf_panel_internal_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex** tau_array, magma_int_t taui, magmaDoubleComplex** dT_array, magma_int_t Ti, magma_int_t Tj, magma_int_t lddt, magmaDoubleComplex** dR_array, magma_int_t Ri, magma_int_t Rj, magma_int_t lddr, magmaDoubleComplex** dwork_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgeqrf_panel_internal_batched(m, n, nb, dA_array, Ai, Aj, ldda, tau_array, taui, dT_array, Ti, Tj, lddt, dR_array, Ri, Rj, lddr, dwork_array, info_array, batchCount, queue)
    ccall((:magma_zgeqrf_panel_internal_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, nb, dA_array, Ai, Aj, ldda, tau_array, taui, dT_array, Ti, Tj, lddt, dR_array, Ri, Rj, lddr, dwork_array, info_array, batchCount, queue)
end

"""
    magma_zgeqrf_panel_batched(m, n, nb, dA_array, ldda, tau_array, dT_array, ldt, dR_array, ldr, dwork_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqrf_panel_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magmaDoubleComplex** dA_array, magma_int_t ldda, magmaDoubleComplex** tau_array, magmaDoubleComplex** dT_array, magma_int_t ldt, magmaDoubleComplex** dR_array, magma_int_t ldr, magmaDoubleComplex** dwork_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgeqrf_panel_batched(m, n, nb, dA_array, ldda, tau_array, dT_array, ldt, dR_array, ldr, dwork_array, info_array, batchCount, queue)
    ccall((:magma_zgeqrf_panel_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, nb, dA_array, ldda, tau_array, dT_array, ldt, dR_array, ldr, dwork_array, info_array, batchCount, queue)
end

"""
    magma_zgels_batched(trans, m, n, nrhs, dA_array, ldda, dB_array, lddb, hwork, lwork, info, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgels_batched( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex **dA_array, magma_int_t ldda, magmaDoubleComplex **dB_array, magma_int_t lddb, magmaDoubleComplex *hwork, magma_int_t lwork, magma_int_t *info, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgels_batched(trans, m, n, nrhs, dA_array, ldda, dB_array, lddb, hwork, lwork, info, batchCount, queue)
    ccall((:magma_zgels_batched, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), trans, m, n, nrhs, dA_array, ldda, dB_array, lddb, hwork, lwork, info, batchCount, queue)
end

"""
    magma_zgeqr2_fused_reg_tall_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqr2_fused_reg_tall_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t* info_array, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgeqr2_fused_reg_tall_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)
    ccall((:magma_zgeqr2_fused_reg_tall_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)
end

"""
    magma_zgeqr2_fused_reg_medium_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqr2_fused_reg_medium_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t* info_array, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgeqr2_fused_reg_medium_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)
    ccall((:magma_zgeqr2_fused_reg_medium_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)
end

"""
    magma_zgeqr2_fused_reg_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqr2_fused_reg_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t* info_array, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgeqr2_fused_reg_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)
    ccall((:magma_zgeqr2_fused_reg_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue)
end

"""
    magma_zgeqr2_fused_sm_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, nthreads, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqr2_fused_sm_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t* info_array, magma_int_t nthreads, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgeqr2_fused_sm_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, nthreads, check_launch_only, batchCount, queue)
    ccall((:magma_zgeqr2_fused_sm_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, nthreads, check_launch_only, batchCount, queue)
end

"""
    magma_zgeqr2_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgeqr2_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgeqr2_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue)
    ccall((:magma_zgeqr2_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue)
end

"""
    magma_zlarf_fused_reg_tall_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarf_fused_reg_tall_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zlarf_fused_reg_tall_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)
    ccall((:magma_zlarf_fused_reg_tall_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)
end

"""
    magma_zlarf_fused_reg_medium_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarf_fused_reg_medium_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zlarf_fused_reg_medium_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)
    ccall((:magma_zlarf_fused_reg_medium_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)
end

"""
    magma_zlarf_fused_reg_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarf_fused_reg_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zlarf_fused_reg_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)
    ccall((:magma_zlarf_fused_reg_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue)
end

"""
    magma_zlarf_fused_sm_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, nthreads, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarf_fused_sm_batched( magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv, magmaDoubleComplex **dtau_array, magma_int_t taui, magma_int_t nthreads, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zlarf_fused_sm_batched(m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, nthreads, check_launch_only, batchCount, queue)
    ccall((:magma_zlarf_fused_sm_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, nb, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, nthreads, check_launch_only, batchCount, queue)
end

"""
    magma_zlarfb_gemm_internal_batched(side, trans, direct, storev, m, n, k, dV_array, vi, vj, lddv, dT_array, Ti, Tj, lddt, dC_array, Ci, Cj, lddc, dwork_array, ldwork, dworkvt_array, ldworkvt, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarfb_gemm_internal_batched( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_const_ptr dV_array[], magma_int_t vi, magma_int_t vj, magma_int_t lddv, magmaDoubleComplex_const_ptr dT_array[], magma_int_t Ti, magma_int_t Tj, magma_int_t lddt, magmaDoubleComplex_ptr dC_array[], magma_int_t Ci, magma_int_t Cj, magma_int_t lddc, magmaDoubleComplex_ptr dwork_array[], magma_int_t ldwork, magmaDoubleComplex_ptr dworkvt_array[], magma_int_t ldworkvt, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlarfb_gemm_internal_batched(side, trans, direct, storev, m, n, k, dV_array, vi, vj, lddv, dT_array, Ti, Tj, lddt, dC_array, Ci, Cj, lddc, dwork_array, ldwork, dworkvt_array, ldworkvt, batchCount, queue)
    ccall((:magma_zlarfb_gemm_internal_batched, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_direct_t, magma_storev_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), side, trans, direct, storev, m, n, k, dV_array, vi, vj, lddv, dT_array, Ti, Tj, lddt, dC_array, Ci, Cj, lddc, dwork_array, ldwork, dworkvt_array, ldworkvt, batchCount, queue)
end

"""
    magma_zlarfb_gemm_batched(side, trans, direct, storev, m, n, k, dV_array, lddv, dT_array, lddt, dC_array, lddc, dwork_array, ldwork, dworkvt_array, ldworkvt, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarfb_gemm_batched( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_const_ptr dV_array[], magma_int_t lddv, magmaDoubleComplex_const_ptr dT_array[], magma_int_t lddt, magmaDoubleComplex_ptr dC_array[], magma_int_t lddc, magmaDoubleComplex_ptr dwork_array[], magma_int_t ldwork, magmaDoubleComplex_ptr dworkvt_array[], magma_int_t ldworkvt, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlarfb_gemm_batched(side, trans, direct, storev, m, n, k, dV_array, lddv, dT_array, lddt, dC_array, lddc, dwork_array, ldwork, dworkvt_array, ldworkvt, batchCount, queue)
    ccall((:magma_zlarfb_gemm_batched, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_direct_t, magma_storev_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex_const_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_queue_t), side, trans, direct, storev, m, n, k, dV_array, lddv, dT_array, lddt, dC_array, lddc, dwork_array, ldwork, dworkvt_array, ldworkvt, batchCount, queue)
end

"""
    magma_zlarft_internal_batched(n, k, stair_T, v_array, vi, vj, ldv, tau_array, taui, T_array, Ti, Tj, ldt, work_array, lwork, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarft_internal_batched( magma_int_t n, magma_int_t k, magma_int_t stair_T, magmaDoubleComplex **v_array, magma_int_t vi, magma_int_t vj, magma_int_t ldv, magmaDoubleComplex **tau_array, magma_int_t taui, magmaDoubleComplex **T_array, magma_int_t Ti, magma_int_t Tj, magma_int_t ldt, magmaDoubleComplex **work_array, magma_int_t lwork, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlarft_internal_batched(n, k, stair_T, v_array, vi, vj, ldv, tau_array, taui, T_array, Ti, Tj, ldt, work_array, lwork, batchCount, queue)
    ccall((:magma_zlarft_internal_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), n, k, stair_T, v_array, vi, vj, ldv, tau_array, taui, T_array, Ti, Tj, ldt, work_array, lwork, batchCount, queue)
end

"""
    magma_zlarft_batched(n, k, stair_T, v_array, ldv, tau_array, T_array, ldt, work_array, lwork, batchCount, queue)


### Prototype
```c
magma_int_t magma_zlarft_batched( magma_int_t n, magma_int_t k, magma_int_t stair_T, magmaDoubleComplex **v_array, magma_int_t ldv, magmaDoubleComplex **tau_array, magmaDoubleComplex **T_array, magma_int_t ldt, magmaDoubleComplex **work_array, magma_int_t lwork, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlarft_batched(n, k, stair_T, v_array, ldv, tau_array, T_array, ldt, work_array, lwork, batchCount, queue)
    ccall((:magma_zlarft_batched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), n, k, stair_T, v_array, ldv, tau_array, T_array, ldt, work_array, lwork, batchCount, queue)
end

"""
    magma_zlarft_sm32x32_batched(n, k, v_array, vi, vj, ldv, tau_array, taui, T_array, Ti, Tj, ldt, batchCount, queue)


### Prototype
```c
void magma_zlarft_sm32x32_batched( magma_int_t n, magma_int_t k, magmaDoubleComplex **v_array, magma_int_t vi, magma_int_t vj, magma_int_t ldv, magmaDoubleComplex **tau_array, magma_int_t taui, magmaDoubleComplex **T_array, magma_int_t Ti, magma_int_t Tj, magma_int_t ldt, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlarft_sm32x32_batched(n, k, v_array, vi, vj, ldv, tau_array, taui, T_array, Ti, Tj, ldt, batchCount, queue)
    ccall((:magma_zlarft_sm32x32_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), n, k, v_array, vi, vj, ldv, tau_array, taui, T_array, Ti, Tj, ldt, batchCount, queue)
end

"""
    magmablas_zlarft_recztrmv_sm32x32(m, n, tau, Trec, ldtrec, Ttri, ldttri, queue)


### Prototype
```c
void magmablas_zlarft_recztrmv_sm32x32( magma_int_t m, magma_int_t n, magmaDoubleComplex *tau, magmaDoubleComplex *Trec, magma_int_t ldtrec, magmaDoubleComplex *Ttri, magma_int_t ldttri, magma_queue_t queue);
```
"""
function magmablas_zlarft_recztrmv_sm32x32(m, n, tau, Trec, ldtrec, Ttri, ldttri, queue)
    ccall((:magmablas_zlarft_recztrmv_sm32x32, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), m, n, tau, Trec, ldtrec, Ttri, ldttri, queue)
end

"""
    magmablas_zlarft_recztrmv_sm32x32_batched(m, n, tau_array, taui, Trec_array, Treci, Trecj, ldtrec, Ttri_array, Ttrii, Ttrij, ldttri, batchCount, queue)


### Prototype
```c
void magmablas_zlarft_recztrmv_sm32x32_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **tau_array, magma_int_t taui, magmaDoubleComplex **Trec_array, magma_int_t Treci, magma_int_t Trecj, magma_int_t ldtrec, magmaDoubleComplex **Ttri_array, magma_int_t Ttrii, magma_int_t Ttrij, magma_int_t ldttri, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zlarft_recztrmv_sm32x32_batched(m, n, tau_array, taui, Trec_array, Treci, Trecj, ldtrec, Ttri_array, Ttrii, Ttrij, ldttri, batchCount, queue)
    ccall((:magmablas_zlarft_recztrmv_sm32x32_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, tau_array, taui, Trec_array, Treci, Trecj, ldtrec, Ttri_array, Ttrii, Ttrij, ldttri, batchCount, queue)
end

"""
    magmablas_zlarft_ztrmv_sm32x32(m, n, tau, Tin, ldtin, Tout, ldtout, queue)


### Prototype
```c
void magmablas_zlarft_ztrmv_sm32x32( magma_int_t m, magma_int_t n, magmaDoubleComplex *tau, magmaDoubleComplex *Tin, magma_int_t ldtin, magmaDoubleComplex *Tout, magma_int_t ldtout, magma_queue_t queue);
```
"""
function magmablas_zlarft_ztrmv_sm32x32(m, n, tau, Tin, ldtin, Tout, ldtout, queue)
    ccall((:magmablas_zlarft_ztrmv_sm32x32, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), m, n, tau, Tin, ldtin, Tout, ldtout, queue)
end

"""
    magmablas_zlarft_ztrmv_sm32x32_batched(m, n, tau_array, taui, Tin_array, Tini, Tinj, ldtin, Tout_array, Touti, Toutj, ldtout, batchCount, queue)


### Prototype
```c
void magmablas_zlarft_ztrmv_sm32x32_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **tau_array, magma_int_t taui, magmaDoubleComplex **Tin_array, magma_int_t Tini, magma_int_t Tinj, magma_int_t ldtin, magmaDoubleComplex **Tout_array, magma_int_t Touti, magma_int_t Toutj, magma_int_t ldtout, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zlarft_ztrmv_sm32x32_batched(m, n, tau_array, taui, Tin_array, Tini, Tinj, ldtin, Tout_array, Touti, Toutj, ldtout, batchCount, queue)
    ccall((:magmablas_zlarft_ztrmv_sm32x32_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), m, n, tau_array, taui, Tin_array, Tini, Tinj, ldtin, Tout_array, Touti, Toutj, ldtout, batchCount, queue)
end

"""
    magmablas_dznrm2_cols_batched(m, n, dA_array, lda, dxnorm_array, batchCount)


### Prototype
```c
void magmablas_dznrm2_cols_batched( magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, double **dxnorm_array, magma_int_t batchCount);
```
"""
function magmablas_dznrm2_cols_batched(m, n, dA_array, lda, dxnorm_array, batchCount)
    ccall((:magmablas_dznrm2_cols_batched, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{Cdouble}}, magma_int_t), m, n, dA_array, lda, dxnorm_array, batchCount)
end

"""
    magma_zlarfgx_batched(n, dx0_array, dx_array, dtau_array, dxnorm_array, dR_array, it, batchCount)


### Prototype
```c
void magma_zlarfgx_batched( magma_int_t n, magmaDoubleComplex **dx0_array, magmaDoubleComplex **dx_array, magmaDoubleComplex **dtau_array, double **dxnorm_array, magmaDoubleComplex **dR_array, magma_int_t it, magma_int_t batchCount);
```
"""
function magma_zlarfgx_batched(n, dx0_array, dx_array, dtau_array, dxnorm_array, dR_array, it, batchCount)
    ccall((:magma_zlarfgx_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{Cdouble}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), n, dx0_array, dx_array, dtau_array, dxnorm_array, dR_array, it, batchCount)
end

"""
    magma_zlarfx_batched_v4(m, n, v_array, tau_array, C_array, ldc, xnorm_array, step, batchCount)


### Prototype
```c
void magma_zlarfx_batched_v4( magma_int_t m, magma_int_t n, magmaDoubleComplex **v_array, magmaDoubleComplex **tau_array, magmaDoubleComplex **C_array, magma_int_t ldc, double **xnorm_array, magma_int_t step, magma_int_t batchCount);
```
"""
function magma_zlarfx_batched_v4(m, n, v_array, tau_array, C_array, ldc, xnorm_array, step, batchCount)
    ccall((:magma_zlarfx_batched_v4, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{Cdouble}}, magma_int_t, magma_int_t), m, n, v_array, tau_array, C_array, ldc, xnorm_array, step, batchCount)
end

"""
    magmablas_zlarfg_batched(n, dalpha_array, dx_array, incx, dtau_array, batchCount)


### Prototype
```c
void magmablas_zlarfg_batched( magma_int_t n, magmaDoubleComplex** dalpha_array, magmaDoubleComplex** dx_array, magma_int_t incx, magmaDoubleComplex** dtau_array, magma_int_t batchCount );
```
"""
function magmablas_zlarfg_batched(n, dalpha_array, dx_array, incx, dtau_array, batchCount)
    ccall((:magmablas_zlarfg_batched, libmagma), Cvoid, (magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t), n, dalpha_array, dx_array, incx, dtau_array, batchCount)
end

"""
    magma_zpotrf_lpout_batched(uplo, n, dA_array, ai, aj, lda, gbstep, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_lpout_batched( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_lpout_batched(uplo, n, dA_array, ai, aj, lda, gbstep, info_array, batchCount, queue)
    ccall((:magma_zpotrf_lpout_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, dA_array, ai, aj, lda, gbstep, info_array, batchCount, queue)
end

"""
    magma_zpotrf_lpin_batched(uplo, n, dA_array, ai, aj, lda, gbstep, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_lpin_batched( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t lda, magma_int_t gbstep, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_lpin_batched(uplo, n, dA_array, ai, aj, lda, gbstep, info_array, batchCount, queue)
    ccall((:magma_zpotrf_lpin_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, dA_array, ai, aj, lda, gbstep, info_array, batchCount, queue)
end

"""
    magma_zpotrf_v33_batched(uplo, n, dA_array, lda, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_v33_batched( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t lda, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_v33_batched(uplo, n, dA_array, lda, info_array, batchCount, queue)
    ccall((:magma_zpotrf_v33_batched, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, dA_array, lda, info_array, batchCount, queue)
end

"""
    blas_zlacpy_batched(uplo, m, n, hA_array, lda, hB_array, ldb, batchCount)

host interface
### Prototype
```c
void blas_zlacpy_batched( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex const * const * hA_array, magma_int_t lda, magmaDoubleComplex **hB_array, magma_int_t ldb, magma_int_t batchCount );
```
"""
function blas_zlacpy_batched(uplo, m, n, hA_array, lda, hB_array, ldb, batchCount)
    ccall((:blas_zlacpy_batched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), uplo, m, n, hA_array, lda, hB_array, ldb, batchCount)
end

"""
    blas_zgemm_batched(transA, transB, m, n, k, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)


### Prototype
```c
void blas_zgemm_batched( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda, magmaDoubleComplex const * const * hB_array, magma_int_t ldb, magmaDoubleComplex beta, magmaDoubleComplex **hC_array, magma_int_t ldc, magma_int_t batchCount );
```
"""
function blas_zgemm_batched(transA, transB, m, n, k, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)
    ccall((:blas_zgemm_batched, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), transA, transB, m, n, k, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)
end

"""
    blas_ztrsm_batched(side, uplo, transA, diag, m, n, alpha, hA_array, lda, hB_array, ldb, batchCount)


### Prototype
```c
void blas_ztrsm_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **hA_array, magma_int_t lda, magmaDoubleComplex **hB_array, magma_int_t ldb, magma_int_t batchCount );
```
"""
function blas_ztrsm_batched(side, uplo, transA, diag, m, n, alpha, hA_array, lda, hB_array, ldb, batchCount)
    ccall((:blas_ztrsm_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), side, uplo, transA, diag, m, n, alpha, hA_array, lda, hB_array, ldb, batchCount)
end

"""
    blas_ztrmm_batched(side, uplo, transA, diag, m, n, alpha, hA_array, lda, hB_array, ldb, batchCount)


### Prototype
```c
void blas_ztrmm_batched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **hA_array, magma_int_t lda, magmaDoubleComplex **hB_array, magma_int_t ldb, magma_int_t batchCount );
```
"""
function blas_ztrmm_batched(side, uplo, transA, diag, m, n, alpha, hA_array, lda, hB_array, ldb, batchCount)
    ccall((:blas_ztrmm_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), side, uplo, transA, diag, m, n, alpha, hA_array, lda, hB_array, ldb, batchCount)
end

"""
    blas_zhemm_batched(side, uplo, m, n, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)


### Prototype
```c
void blas_zhemm_batched( magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex **hA_array, magma_int_t lda, magmaDoubleComplex **hB_array, magma_int_t ldb, magmaDoubleComplex beta, magmaDoubleComplex **hC_array, magma_int_t ldc, magma_int_t batchCount );
```
"""
function blas_zhemm_batched(side, uplo, m, n, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)
    ccall((:blas_zhemm_batched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), side, uplo, m, n, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)
end

"""
    blas_zherk_batched(uplo, trans, n, k, alpha, hA_array, lda, beta, hC_array, ldc, batchCount)


### Prototype
```c
void blas_zherk_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, double alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda, double beta, magmaDoubleComplex **hC_array, magma_int_t ldc, magma_int_t batchCount );
```
"""
function blas_zherk_batched(uplo, trans, n, k, alpha, hA_array, lda, beta, hC_array, ldc, batchCount)
    ccall((:blas_zherk_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), uplo, trans, n, k, alpha, hA_array, lda, beta, hC_array, ldc, batchCount)
end

"""
    blas_zher2k_batched(uplo, trans, n, k, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)


### Prototype
```c
void blas_zher2k_batched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * hA_array, magma_int_t lda, magmaDoubleComplex const * const * hB_array, magma_int_t ldb, double beta, magmaDoubleComplex **hC_array, magma_int_t ldc, magma_int_t batchCount );
```
"""
function blas_zher2k_batched(uplo, trans, n, k, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)
    ccall((:blas_zher2k_batched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t), uplo, trans, n, k, alpha, hA_array, lda, hB_array, ldb, beta, hC_array, ldc, batchCount)
end

"""
    zset_stepinit_ipiv(ipiv_array, pm, batchCount)

for debugging purpose
### Prototype
```c
void zset_stepinit_ipiv( magma_int_t **ipiv_array, magma_int_t pm, magma_int_t batchCount);
```
"""
function zset_stepinit_ipiv(ipiv_array, pm, batchCount)
    ccall((:zset_stepinit_ipiv, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t), ipiv_array, pm, batchCount)
end

"""
    magma_hset_pointer(output_array, input, lda, row, column, batch_offset, batchCount, queue)


### Prototype
```c
void magma_hset_pointer( magmaHalf **output_array, magmaHalf *input, magma_int_t lda, magma_int_t row, magma_int_t column, magma_int_t batch_offset, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_hset_pointer(output_array, input, lda, row, column, batch_offset, batchCount, queue)
    ccall((:magma_hset_pointer, libmagma), Cvoid, (Ptr{Ptr{magmaHalf}}, Ptr{magmaHalf}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input, lda, row, column, batch_offset, batchCount, queue)
end

"""
    magmablas_hgemm_batched(transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batchCount, queue)


### Prototype
```c
magma_int_t magmablas_hgemm_batched( magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k, magmaHalf alpha, magmaHalf const * const * dAarray, magma_int_t ldda, magmaHalf const * const * dBarray, magma_int_t lddb, magmaHalf beta, magmaHalf **dCarray, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_hgemm_batched(transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batchCount, queue)
    ccall((:magmablas_hgemm_batched, libmagma), magma_int_t, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaHalf, Ptr{Ptr{magmaHalf}}, magma_int_t, Ptr{Ptr{magmaHalf}}, magma_int_t, magmaHalf, Ptr{Ptr{magmaHalf}}, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dAarray, ldda, dBarray, lddb, beta, dCarray, lddc, batchCount, queue)
end

"""
    setup_pivinfo_batched(pivinfo_array, ipiv_array, ipiv_offset, m, nb, batchCount, queue)


### Prototype
```c
void setup_pivinfo_batched( magma_int_t **pivinfo_array, magma_int_t **ipiv_array, magma_int_t ipiv_offset, magma_int_t m, magma_int_t nb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function setup_pivinfo_batched(pivinfo_array, ipiv_array, ipiv_offset, m, nb, batchCount, queue)
    ccall((:setup_pivinfo_batched, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), pivinfo_array, ipiv_array, ipiv_offset, m, nb, batchCount, queue)
end

"""
    adjust_ipiv_batched(ipiv_array, ipiv_offset, m, offset, batchCount, queue)


### Prototype
```c
void adjust_ipiv_batched( magma_int_t **ipiv_array, magma_int_t ipiv_offset, magma_int_t m, magma_int_t offset, magma_int_t batchCount, magma_queue_t queue);
```
"""
function adjust_ipiv_batched(ipiv_array, ipiv_offset, m, offset, batchCount, queue)
    ccall((:adjust_ipiv_batched, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), ipiv_array, ipiv_offset, m, offset, batchCount, queue)
end

"""
    magma_idisplace_pointers(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_idisplace_pointers(magma_int_t **output_array, magma_int_t **input_array, magma_int_t lda, magma_int_t row, magma_int_t column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_idisplace_pointers(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_idisplace_pointers, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    stepinit_ipiv(ipiv_array, pm, batchCount, queue)


### Prototype
```c
void stepinit_ipiv(magma_int_t **ipiv_array, magma_int_t pm, magma_int_t batchCount, magma_queue_t queue);
```
"""
function stepinit_ipiv(ipiv_array, pm, batchCount, queue)
    ccall((:stepinit_ipiv, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_queue_t), ipiv_array, pm, batchCount, queue)
end

"""
    adjust_ipiv(ipiv, m, offset, queue)


### Prototype
```c
void adjust_ipiv( magma_int_t *ipiv, magma_int_t m, magma_int_t offset, magma_queue_t queue);
```
"""
function adjust_ipiv(ipiv, m, offset, queue)
    ccall((:adjust_ipiv, libmagma), Cvoid, (Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), ipiv, m, offset, queue)
end

"""
    magma_iset_pointer(output_array, input, lda, row, column, batchSize, batchCount, queue)


### Prototype
```c
void magma_iset_pointer( magma_int_t **output_array, magma_int_t *input, magma_int_t lda, magma_int_t row, magma_int_t column, magma_int_t batchSize, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_iset_pointer(output_array, input, lda, row, column, batchSize, batchCount, queue)
    ccall((:magma_iset_pointer, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input, lda, row, column, batchSize, batchCount, queue)
end

"""
    magma_get_zgetrf_vbatched_nbparam(max_m, max_n, nb, recnb)

 control and tuning
### Prototype
```c
void magma_get_zgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb);
```
"""
function magma_get_zgetrf_vbatched_nbparam(max_m, max_n, nb, recnb)
    ccall((:magma_get_zgetrf_vbatched_nbparam, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), max_m, max_n, nb, recnb)
end

"""
    magma_zgetf2_fused_vbatched(max_M, max_N, max_minMN, max_MxN, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue)

 LAPACK vbatched routines
### Prototype
```c
magma_int_t magma_zgetf2_fused_vbatched( magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN, magma_int_t* M, magma_int_t* N, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magma_int_t **dipiv_array, magma_int_t ipiv_i, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetf2_fused_vbatched(max_M, max_N, max_minMN, max_MxN, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue)
    ccall((:magma_zgetf2_fused_vbatched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), max_M, max_N, max_minMN, max_MxN, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue)
end

"""
    magma_zgetf2_fused_sm_vbatched(max_M, max_N, max_minMN, max_MxN, m, n, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, gbstep, nthreads, check_launch_only, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetf2_fused_sm_vbatched( magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN, magma_int_t* m, magma_int_t* n, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magma_int_t** dipiv_array, magma_int_t ipiv_i, magma_int_t* info_array, magma_int_t gbstep, magma_int_t nthreads, magma_int_t check_launch_only, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_zgetf2_fused_sm_vbatched(max_M, max_N, max_minMN, max_MxN, m, n, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, gbstep, nthreads, check_launch_only, batchCount, queue)
    ccall((:magma_zgetf2_fused_sm_vbatched, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), max_M, max_N, max_minMN, max_MxN, m, n, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, gbstep, nthreads, check_launch_only, batchCount, queue)
end

"""
    magma_zgetrf_vbatched(m, n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_vbatched( magma_int_t* m, magma_int_t* n, magmaDoubleComplex **dA_array, magma_int_t *ldda, magma_int_t **ipiv_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_vbatched(m, n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)
    ccall((:magma_zgetrf_vbatched, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, dA_array, ldda, ipiv_array, info_array, batchCount, queue)
end

"""
    magma_zgetrf_vbatched_max_nocheck(m, n, minmn, max_m, max_n, max_minmn, max_mxn, nb, recnb, dA_array, ldda, ipiv_array, pivinfo_array, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_vbatched_max_nocheck( magma_int_t* m, magma_int_t* n, magma_int_t* minmn, magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn, magma_int_t nb, magma_int_t recnb, magmaDoubleComplex **dA_array, magma_int_t *ldda, magma_int_t **ipiv_array, magma_int_t** pivinfo_array, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_vbatched_max_nocheck(m, n, minmn, max_m, max_n, max_minmn, max_mxn, nb, recnb, dA_array, ldda, ipiv_array, pivinfo_array, info_array, batchCount, queue)
    ccall((:magma_zgetrf_vbatched_max_nocheck, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, minmn, max_m, max_n, max_minmn, max_mxn, nb, recnb, dA_array, ldda, ipiv_array, pivinfo_array, info_array, batchCount, queue)
end

"""
    magma_zgetrf_vbatched_max_nocheck_work(m, n, max_m, max_n, max_minmn, max_mxn, dA_array, ldda, dipiv_array, info_array, work, lwork, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_vbatched_max_nocheck_work( magma_int_t* m, magma_int_t* n, magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn, magmaDoubleComplex **dA_array, magma_int_t *ldda, magma_int_t **dipiv_array, magma_int_t *info_array, void* work, magma_int_t* lwork, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_vbatched_max_nocheck_work(m, n, max_m, max_n, max_minmn, max_mxn, dA_array, ldda, dipiv_array, info_array, work, lwork, batchCount, queue)
    ccall((:magma_zgetrf_vbatched_max_nocheck_work, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, Ptr{Cvoid}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, max_m, max_n, max_minmn, max_mxn, dA_array, ldda, dipiv_array, info_array, work, lwork, batchCount, queue)
end

"""
    magma_izamax_vbatched(length, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_i, info_array, step, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_izamax_vbatched( magma_int_t length, magma_int_t *M, magma_int_t *N, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magma_int_t** ipiv_array, magma_int_t ipiv_i, magma_int_t *info_array, magma_int_t step, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_izamax_vbatched(length, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_i, info_array, step, gbstep, batchCount, queue)
    ccall((:magma_izamax_vbatched, libmagma), magma_int_t, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), length, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_i, info_array, step, gbstep, batchCount, queue)
end

"""
    magma_zswap_vbatched(max_n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, piv_adjustment, batchCount, queue)


### Prototype
```c
magma_int_t magma_zswap_vbatched( magma_int_t max_n, magma_int_t *M, magma_int_t *N, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda, magma_int_t** ipiv_array, magma_int_t piv_adjustment, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zswap_vbatched(max_n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, piv_adjustment, batchCount, queue)
    ccall((:magma_zswap_vbatched, libmagma), magma_int_t, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_queue_t), max_n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, piv_adjustment, batchCount, queue)
end

"""
    magma_zscal_zgeru_vbatched(max_M, max_N, M, N, dA_array, Ai, Aj, ldda, info_array, step, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zscal_zgeru_vbatched( magma_int_t max_M, magma_int_t max_N, magma_int_t *M, magma_int_t *N, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda, magma_int_t *info_array, magma_int_t step, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zscal_zgeru_vbatched(max_M, max_N, M, N, dA_array, Ai, Aj, ldda, info_array, step, gbstep, batchCount, queue)
    ccall((:magma_zscal_zgeru_vbatched, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), max_M, max_N, M, N, dA_array, Ai, Aj, ldda, info_array, step, gbstep, batchCount, queue)
end

"""
    magma_zgetf2_vbatched(m, n, minmn, max_m, max_n, max_minmn, max_mxn, dA_array, Ai, Aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetf2_vbatched( magma_int_t *m, magma_int_t *n, magma_int_t *minmn, magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda, magma_int_t **ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetf2_vbatched(m, n, minmn, max_m, max_n, max_minmn, max_mxn, dA_array, Ai, Aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetf2_vbatched, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, minmn, max_m, max_n, max_minmn, max_mxn, dA_array, Ai, Aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue)
end

"""
    magma_zgetrf_recpanel_vbatched(m, n, minmn, max_m, max_n, max_minmn, max_mxn, min_recpnb, dA_array, Ai, Aj, ldda, dipiv_array, dipiv_i, dpivinfo_array, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zgetrf_recpanel_vbatched( magma_int_t* m, magma_int_t* n, magma_int_t* minmn, magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn, magma_int_t min_recpnb, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zgetrf_recpanel_vbatched(m, n, minmn, max_m, max_n, max_minmn, max_mxn, min_recpnb, dA_array, Ai, Aj, ldda, dipiv_array, dipiv_i, dpivinfo_array, info_array, gbstep, batchCount, queue)
    ccall((:magma_zgetrf_recpanel_vbatched, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{Ptr{magma_int_t}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), m, n, minmn, max_m, max_n, max_minmn, max_mxn, min_recpnb, dA_array, Ai, Aj, ldda, dipiv_array, dipiv_i, dpivinfo_array, info_array, gbstep, batchCount, queue)
end

"""
    magma_zlaswp_left_rowserial_vbatched(n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2, batchCount, queue)


### Prototype
```c
void magma_zlaswp_left_rowserial_vbatched( magma_int_t n, magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda, magma_int_t **ipiv_array, magma_int_t ipiv_offset, magma_int_t k1, magma_int_t k2, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_left_rowserial_vbatched(n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2, batchCount, queue)
    ccall((:magma_zlaswp_left_rowserial_vbatched, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2, batchCount, queue)
end

"""
    magma_zlaswp_right_rowserial_vbatched(n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2, batchCount, queue)


### Prototype
```c
void magma_zlaswp_right_rowserial_vbatched( magma_int_t n, magma_int_t *M, magma_int_t *N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda, magma_int_t **ipiv_array, magma_int_t ipiv_offset, magma_int_t k1, magma_int_t k2, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_right_rowserial_vbatched(n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2, batchCount, queue)
    ccall((:magma_zlaswp_right_rowserial_vbatched, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_offset, k1, k2, batchCount, queue)
end

"""
    magma_zlaswp_left_rowparallel_vbatched(n, M, N, dA_array, Ai, Aj, ldda, k1, k2, pivinfo_array, pivinfo_i, batchCount, queue)


### Prototype
```c
void magma_zlaswp_left_rowparallel_vbatched( magma_int_t n, magma_int_t* M, magma_int_t* N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magma_int_t k1, magma_int_t k2, magma_int_t **pivinfo_array, magma_int_t pivinfo_i, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_left_rowparallel_vbatched(n, M, N, dA_array, Ai, Aj, ldda, k1, k2, pivinfo_array, pivinfo_i, batchCount, queue)
    ccall((:magma_zlaswp_left_rowparallel_vbatched, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_queue_t), n, M, N, dA_array, Ai, Aj, ldda, k1, k2, pivinfo_array, pivinfo_i, batchCount, queue)
end

"""
    magma_zlaswp_right_rowparallel_vbatched(n, M, N, dA_array, Ai, Aj, ldda, k1, k2, pivinfo_array, pivinfo_i, batchCount, queue)


### Prototype
```c
void magma_zlaswp_right_rowparallel_vbatched( magma_int_t n, magma_int_t* M, magma_int_t* N, magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magma_int_t k1, magma_int_t k2, magma_int_t **pivinfo_array, magma_int_t pivinfo_i, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zlaswp_right_rowparallel_vbatched(n, M, N, dA_array, Ai, Aj, ldda, k1, k2, pivinfo_array, pivinfo_i, batchCount, queue)
    ccall((:magma_zlaswp_right_rowparallel_vbatched, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, magma_int_t, magma_queue_t), n, M, N, dA_array, Ai, Aj, ldda, k1, k2, pivinfo_array, pivinfo_i, batchCount, queue)
end

"""
    magma_zpotrf_lpout_vbatched(uplo, n, max_n, dA_array, lda, gbstep, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_lpout_vbatched( magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n, magmaDoubleComplex **dA_array, magma_int_t *lda, magma_int_t gbstep, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_lpout_vbatched(uplo, n, max_n, dA_array, lda, gbstep, info_array, batchCount, queue)
    ccall((:magma_zpotrf_lpout_vbatched, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, max_n, dA_array, lda, gbstep, info_array, batchCount, queue)
end

"""
    magma_zpotf2_vbatched(uplo, n, max_n, dA_array, lda, dA_displ, dW_displ, dB_displ, dC_displ, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotf2_vbatched( magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, magmaDoubleComplex **dA_array, magma_int_t* lda, magmaDoubleComplex **dA_displ, magmaDoubleComplex **dW_displ, magmaDoubleComplex **dB_displ, magmaDoubleComplex **dC_displ, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotf2_vbatched(uplo, n, max_n, dA_array, lda, dA_displ, dW_displ, dB_displ, dC_displ, info_array, gbstep, batchCount, queue)
    ccall((:magma_zpotf2_vbatched, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, max_n, dA_array, lda, dA_displ, dW_displ, dB_displ, dC_displ, info_array, gbstep, batchCount, queue)
end

"""
    magma_zpotrf_panel_vbatched(uplo, n, max_n, ibvec, nb, dA_array, ldda, dX_array, dX_length, dinvA_array, dinvA_length, dW0_displ, dW1_displ, dW2_displ, dW3_displ, dW4_displ, info_array, gbstep, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_panel_vbatched( magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, magma_int_t *ibvec, magma_int_t nb, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dX_array, magma_int_t* dX_length, magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length, magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ, magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ, magmaDoubleComplex** dW4_displ, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_panel_vbatched(uplo, n, max_n, ibvec, nb, dA_array, ldda, dX_array, dX_length, dinvA_array, dinvA_length, dW0_displ, dW1_displ, dW2_displ, dW3_displ, dW4_displ, info_array, gbstep, batchCount, queue)
    ccall((:magma_zpotrf_panel_vbatched, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, max_n, ibvec, nb, dA_array, ldda, dX_array, dX_length, dinvA_array, dinvA_length, dW0_displ, dW1_displ, dW2_displ, dW3_displ, dW4_displ, info_array, gbstep, batchCount, queue)
end

"""
    magma_zpotrf_vbatched_max_nocheck(uplo, n, dA_array, ldda, info_array, batchCount, max_n, queue)


### Prototype
```c
magma_int_t magma_zpotrf_vbatched_max_nocheck( magma_uplo_t uplo, magma_int_t *n, magmaDoubleComplex **dA_array, magma_int_t *ldda, magma_int_t *info_array, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue);
```
"""
function magma_zpotrf_vbatched_max_nocheck(uplo, n, dA_array, ldda, info_array, batchCount, max_n, queue)
    ccall((:magma_zpotrf_vbatched_max_nocheck, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, dA_array, ldda, info_array, batchCount, max_n, queue)
end

"""
    magma_zpotrf_vbatched(uplo, n, dA_array, ldda, info_array, batchCount, queue)


### Prototype
```c
magma_int_t magma_zpotrf_vbatched( magma_uplo_t uplo, magma_int_t *n, magmaDoubleComplex **dA_array, magma_int_t *ldda, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zpotrf_vbatched(uplo, n, dA_array, ldda, info_array, batchCount, queue)
    ccall((:magma_zpotrf_vbatched, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, dA_array, ldda, info_array, batchCount, queue)
end

"""
    magmablas_zgemm_vbatched_core(transA, transB, max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)

 BLAS vbatched routines
/
/* Level 3 
### Prototype
```c
void magmablas_zgemm_vbatched_core( magma_trans_t transA, magma_trans_t transB, magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, magma_int_t* m, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_vbatched_core(transA, transB, max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)
    ccall((:magmablas_zgemm_vbatched_core, libmagma), Cvoid, (magma_trans_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), transA, transB, max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue)
end

"""
    magmablas_zgemm_vbatched_max_nocheck(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, max_k, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_vbatched_max_nocheck( magma_trans_t transA, magma_trans_t transB, magma_int_t* m, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_vbatched_max_nocheck(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, max_k, batchCount, queue)
    ccall((:magmablas_zgemm_vbatched_max_nocheck, libmagma), Cvoid, (magma_trans_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, max_k, batchCount, queue)
end

"""
    magmablas_zgemm_vbatched_max(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, max_k, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_vbatched_max( magma_trans_t transA, magma_trans_t transB, magma_int_t* m, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_vbatched_max(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, max_k, batchCount, queue)
    ccall((:magmablas_zgemm_vbatched_max, libmagma), Cvoid, (magma_trans_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, max_k, batchCount, queue)
end

"""
    magmablas_zgemm_vbatched_nocheck(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_vbatched_nocheck( magma_trans_t transA, magma_trans_t transB, magma_int_t* m, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_vbatched_nocheck(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zgemm_vbatched_nocheck, libmagma), Cvoid, (magma_trans_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zgemm_vbatched(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zgemm_vbatched( magma_trans_t transA, magma_trans_t transB, magma_int_t* m, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zgemm_vbatched(transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zgemm_vbatched, libmagma), Cvoid, (magma_trans_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), transA, transB, m, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zherk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue)


### Prototype
```c
void magmablas_zherk_internal_vbatched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t max_n, magma_int_t max_k, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zherk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue)
    ccall((:magmablas_zherk_internal_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue)
end

"""
    magmablas_zsyrk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue)


### Prototype
```c
void magmablas_zsyrk_internal_vbatched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t max_n, magma_int_t max_k, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyrk_internal_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue)
    ccall((:magmablas_zsyrk_internal_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_n, max_k, batchCount, queue)
end

"""
    magmablas_zherk_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zherk_vbatched_max_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, double alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zherk_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zherk_vbatched_max_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zherk_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zherk_vbatched_max( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, double alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zherk_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zherk_vbatched_max, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zherk_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zherk_vbatched_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, double alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zherk_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zherk_vbatched_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zherk_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zherk_vbatched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, double alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zherk_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zherk_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zsyrk_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zsyrk_vbatched_max_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zsyrk_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zsyrk_vbatched_max_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zsyrk_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zsyrk_vbatched_max( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zsyrk_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zsyrk_vbatched_max, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zsyrk_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyrk_vbatched_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyrk_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zsyrk_vbatched_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zsyrk_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyrk_vbatched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyrk_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zsyrk_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zher2k_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zher2k_vbatched_max_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zher2k_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zher2k_vbatched_max_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zher2k_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zher2k_vbatched_max( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zher2k_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zher2k_vbatched_max, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zher2k_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zher2k_vbatched_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zher2k_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zher2k_vbatched_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zher2k_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zher2k_vbatched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, double beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zher2k_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zher2k_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Cdouble, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zsyr2k_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zsyr2k_vbatched_max_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zsyr2k_vbatched_max_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zsyr2k_vbatched_max_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zsyr2k_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)


### Prototype
```c
void magmablas_zsyr2k_vbatched_max( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
```
"""
function magmablas_zsyr2k_vbatched_max(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
    ccall((:magmablas_zsyr2k_vbatched_max, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_n, max_k, queue)
end

"""
    magmablas_zsyr2k_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyr2k_vbatched_nocheck( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyr2k_vbatched_nocheck(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zsyr2k_vbatched_nocheck, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zsyr2k_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zsyr2k_vbatched( magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k, magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, magma_int_t* ldda, magmaDoubleComplex const * const * dB_array, magma_int_t* lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zsyr2k_vbatched(uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zsyr2k_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_ztrmm_vbatched_core(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_vbatched_core( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_vbatched_core(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_vbatched_core, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_ztrmm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_vbatched_max_nocheck( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_vbatched_max_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrmm_vbatched_max(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_vbatched_max( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_vbatched_max(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_vbatched_max, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrmm_vbatched_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_vbatched_nocheck( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_vbatched_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_vbatched_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrmm_vbatched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrmm_vbatched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrmm_vbatched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrmm_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_small_vbatched(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_small_vbatched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrsm_small_vbatched(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_small_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_vbatched_core(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_vbatched_core( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrsm_vbatched_core(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_vbatched_core, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_vbatched_max_nocheck( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda, magmaDoubleComplex **dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_ztrsm_vbatched_max_nocheck(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_vbatched_max_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_vbatched_max(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_vbatched_max( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_vbatched_max(side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_vbatched_max, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, max_m, max_n, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_vbatched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_vbatched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_vbatched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_inv_outofplace_vbatched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_ztrsm_inv_outofplace_vbatched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t flag, magma_int_t *m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magmaDoubleComplex** dX_array, magma_int_t* lddx, magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length, magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ, magma_int_t resetozero, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_outofplace_vbatched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, max_m, max_n, queue)
    ccall((:magmablas_ztrsm_inv_outofplace_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, max_m, max_n, queue)
end

"""
    magmablas_ztrsm_inv_work_vbatched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_ztrsm_inv_work_vbatched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t flag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magmaDoubleComplex** dX_array, magma_int_t* lddx, magmaDoubleComplex** dinvA_array, magma_int_t* dinvA_length, magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ, magma_int_t resetozero, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_work_vbatched(side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, max_m, max_n, queue)
    ccall((:magmablas_ztrsm_inv_work_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, flag, m, n, alpha, dA_array, ldda, dB_array, lddb, dX_array, lddx, dinvA_array, dinvA_length, dA_displ, dB_displ, dX_displ, dinvA_displ, resetozero, batchCount, max_m, max_n, queue)
end

"""
    magmablas_ztrsm_inv_vbatched_max_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_ztrsm_inv_vbatched_max_nocheck( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_vbatched_max_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, max_m, max_n, queue)
    ccall((:magmablas_ztrsm_inv_vbatched_max_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, max_m, max_n, queue)
end

"""
    magmablas_ztrsm_inv_vbatched_max(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_ztrsm_inv_vbatched_max( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_vbatched_max(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, max_m, max_n, queue)
    ccall((:magmablas_ztrsm_inv_vbatched_max, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, max_m, max_n, queue)
end

"""
    magmablas_ztrsm_inv_vbatched_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_inv_vbatched_nocheck( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_vbatched_nocheck(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_inv_vbatched_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrsm_inv_vbatched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)


### Prototype
```c
void magmablas_ztrsm_inv_vbatched( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex** dA_array, magma_int_t* ldda, magmaDoubleComplex** dB_array, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrsm_inv_vbatched(side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
    ccall((:magmablas_ztrsm_inv_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, alpha, dA_array, ldda, dB_array, lddb, batchCount, queue)
end

"""
    magmablas_ztrtri_diag_vbatched(uplo, diag, nmax, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue)


### Prototype
```c
void magmablas_ztrtri_diag_vbatched( magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n, magmaDoubleComplex const * const *dA_array, magma_int_t *ldda, magmaDoubleComplex **dinvA_array, magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_ztrtri_diag_vbatched(uplo, diag, nmax, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue)
    ccall((:magmablas_ztrtri_diag_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_diag_t, magma_int_t, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, magma_int_t, magma_int_t, magma_queue_t), uplo, diag, nmax, n, dA_array, ldda, dinvA_array, resetozero, batchCount, queue)
end

"""
    magmablas_zhemm_vbatched_core(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN, batchCount, queue)


### Prototype
```c
void magmablas_zhemm_vbatched_core( magma_side_t side, magma_uplo_t uplo, magma_int_t *m, magma_int_t *n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t *ldda, magmaDoubleComplex **dB_array, magma_int_t *lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t *lddc, magma_int_t max_m, magma_int_t max_n, magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, magma_int_t specM, magma_int_t specN, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemm_vbatched_core(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN, batchCount, queue)
    ccall((:magmablas_zhemm_vbatched_core, libmagma), Cvoid, (magma_side_t, magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, specM, specN, batchCount, queue)
end

"""
    magmablas_zhemm_vbatched_max_nocheck(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_zhemm_vbatched_max_nocheck( magma_side_t side, magma_uplo_t uplo, magma_int_t *m, magma_int_t *n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t *ldda, magmaDoubleComplex **dB_array, magma_int_t *lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t *lddc, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue );
```
"""
function magmablas_zhemm_vbatched_max_nocheck(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_m, max_n, queue)
    ccall((:magmablas_zhemm_vbatched_max_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_m, max_n, queue)
end

"""
    magmablas_zhemm_vbatched_max(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_zhemm_vbatched_max( magma_side_t side, magma_uplo_t uplo, magma_int_t *m, magma_int_t *n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t *ldda, magmaDoubleComplex **dB_array, magma_int_t *lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t *lddc, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue );
```
"""
function magmablas_zhemm_vbatched_max(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_m, max_n, queue)
    ccall((:magmablas_zhemm_vbatched_max, libmagma), Cvoid, (magma_side_t, magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, max_m, max_n, queue)
end

"""
    magmablas_zhemm_vbatched_nocheck(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zhemm_vbatched_nocheck( magma_side_t side, magma_uplo_t uplo, magma_int_t *m, magma_int_t *n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t *ldda, magmaDoubleComplex **dB_array, magma_int_t *lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t *lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemm_vbatched_nocheck(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zhemm_vbatched_nocheck, libmagma), Cvoid, (magma_side_t, magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zhemm_vbatched(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)


### Prototype
```c
void magmablas_zhemm_vbatched( magma_side_t side, magma_uplo_t uplo, magma_int_t *m, magma_int_t *n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t *ldda, magmaDoubleComplex **dB_array, magma_int_t *lddb, magmaDoubleComplex beta, magmaDoubleComplex **dC_array, magma_int_t *lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemm_vbatched(side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
    ccall((:magmablas_zhemm_vbatched, libmagma), Cvoid, (magma_side_t, magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, m, n, alpha, dA_array, ldda, dB_array, lddb, beta, dC_array, lddc, batchCount, queue)
end

"""
    magmablas_zgemv_vbatched_max_nocheck(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_m, max_n, queue)

Level 2 
### Prototype
```c
void magmablas_zgemv_vbatched_max_nocheck( magma_trans_t trans, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_zgemv_vbatched_max_nocheck(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_m, max_n, queue)
    ccall((:magmablas_zgemv_vbatched_max_nocheck, libmagma), Cvoid, (magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_m, max_n, queue)
end

"""
    magmablas_zgemv_vbatched_max(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_m, max_n, queue)


### Prototype
```c
void magmablas_zgemv_vbatched_max( magma_trans_t trans, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_zgemv_vbatched_max(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_m, max_n, queue)
    ccall((:magmablas_zgemv_vbatched_max, libmagma), Cvoid, (magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_m, max_n, queue)
end

"""
    magmablas_zgemv_vbatched_nocheck(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zgemv_vbatched_nocheck( magma_trans_t trans, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zgemv_vbatched_nocheck(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
    ccall((:magmablas_zgemv_vbatched_nocheck, libmagma), Cvoid, (magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
end

"""
    magmablas_zgemv_vbatched(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zgemv_vbatched( magma_trans_t trans, magma_int_t* m, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zgemv_vbatched(trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
    ccall((:magmablas_zgemv_vbatched, libmagma), Cvoid, (magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
end

"""
    magmablas_zhemv_vbatched_max_nocheck(uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, max_n, batchCount, queue)


### Prototype
```c
void magmablas_zhemv_vbatched_max_nocheck( magma_uplo_t uplo, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, magma_int_t* ldda, magmaDoubleComplex **dX_array, magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex **dY_array, magma_int_t* incy, magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zhemv_vbatched_max_nocheck(uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, max_n, batchCount, queue)
    ccall((:magmablas_zhemv_vbatched_max_nocheck, libmagma), Cvoid, (magma_uplo_t, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, alpha, dA_array, ldda, dX_array, incx, beta, dY_array, incy, max_n, batchCount, queue)
end

"""
    magmablas_zhemv_vbatched_max(uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_n, queue)


### Prototype
```c
void magmablas_zhemv_vbatched_max( magma_uplo_t uplo, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_int_t max_n, magma_queue_t queue);
```
"""
function magmablas_zhemv_vbatched_max(uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_n, queue)
    ccall((:magmablas_zhemv_vbatched_max, libmagma), Cvoid, (magma_uplo_t, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, max_n, queue)
end

"""
    magmablas_zhemv_vbatched_nocheck(uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zhemv_vbatched_nocheck( magma_uplo_t uplo, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zhemv_vbatched_nocheck(uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
    ccall((:magmablas_zhemv_vbatched_nocheck, libmagma), Cvoid, (magma_uplo_t, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
end

"""
    magmablas_zhemv_vbatched(uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)


### Prototype
```c
void magmablas_zhemv_vbatched( magma_uplo_t uplo, magma_int_t* n, magmaDoubleComplex alpha, magmaDoubleComplex_ptr dA_array[], magma_int_t* ldda, magmaDoubleComplex_ptr dx_array[], magma_int_t* incx, magmaDoubleComplex beta, magmaDoubleComplex_ptr dy_array[], magma_int_t* incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zhemv_vbatched(uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
    ccall((:magmablas_zhemv_vbatched, libmagma), Cvoid, (magma_uplo_t, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue)
end

"""
    magma_zset_pointer_var_cc(output_array, input, lda, row, column, batch_offset, batchCount, queue)

Level 1 */
/* Auxiliary routines 
### Prototype
```c
void magma_zset_pointer_var_cc( magmaDoubleComplex **output_array, magmaDoubleComplex *input, magma_int_t *lda, magma_int_t row, magma_int_t column, magma_int_t *batch_offset, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zset_pointer_var_cc(output_array, input, lda, row, column, batch_offset, batchCount, queue)
    ccall((:magma_zset_pointer_var_cc, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), output_array, input, lda, row, column, batch_offset, batchCount, queue)
end

"""
    magma_zdisplace_pointers_var_cc(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_zdisplace_pointers_var_cc(magmaDoubleComplex **output_array, magmaDoubleComplex **input_array, magma_int_t* lda, magma_int_t row, magma_int_t column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zdisplace_pointers_var_cc(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_zdisplace_pointers_var_cc, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    magma_zdisplace_pointers_var_cv(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_zdisplace_pointers_var_cv(magmaDoubleComplex **output_array, magmaDoubleComplex **input_array, magma_int_t* lda, magma_int_t row, magma_int_t* column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zdisplace_pointers_var_cv(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_zdisplace_pointers_var_cv, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    magma_zdisplace_pointers_var_vc(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_zdisplace_pointers_var_vc(magmaDoubleComplex **output_array, magmaDoubleComplex **input_array, magma_int_t* lda, magma_int_t *row, magma_int_t column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zdisplace_pointers_var_vc(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_zdisplace_pointers_var_vc, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    magma_zdisplace_pointers_var_vv(output_array, input_array, lda, row, column, batchCount, queue)


### Prototype
```c
void magma_zdisplace_pointers_var_vv(magmaDoubleComplex **output_array, magmaDoubleComplex **input_array, magma_int_t* lda, magma_int_t* row, magma_int_t* column, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_zdisplace_pointers_var_vv(output_array, input_array, lda, row, column, batchCount, queue)
    ccall((:magma_zdisplace_pointers_var_vv, libmagma), Cvoid, (Ptr{Ptr{magmaDoubleComplex}}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), output_array, input_array, lda, row, column, batchCount, queue)
end

"""
    magmablas_zlaset_vbatched(uplo, max_m, max_n, m, n, offdiag, diag, dAarray, ldda, batchCount, queue)


### Prototype
```c
void magmablas_zlaset_vbatched( magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex offdiag, magmaDoubleComplex diag, magmaDoubleComplex_ptr dAarray[], magma_int_t* ldda, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magmablas_zlaset_vbatched(uplo, max_m, max_n, m, n, offdiag, diag, dAarray, ldda, batchCount, queue)
    ccall((:magmablas_zlaset_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magmaDoubleComplex, magmaDoubleComplex, Ptr{magmaDoubleComplex_ptr}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, max_m, max_n, m, n, offdiag, diag, dAarray, ldda, batchCount, queue)
end

"""
    magmablas_zlacpy_vbatched(uplo, max_m, max_n, m, n, dAarray, ldda, dBarray, lddb, batchCount, queue)


### Prototype
```c
void magmablas_zlacpy_vbatched( magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n, magmaDoubleComplex const * const * dAarray, magma_int_t* ldda, magmaDoubleComplex** dBarray, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magmablas_zlacpy_vbatched(uplo, max_m, max_n, m, n, dAarray, ldda, dBarray, lddb, batchCount, queue)
    ccall((:magmablas_zlacpy_vbatched, libmagma), Cvoid, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, Ptr{Ptr{magmaDoubleComplex}}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, max_m, max_n, m, n, dAarray, ldda, dBarray, lddb, batchCount, queue)
end

# no prototype is found for this function at magma_zvbatched.h:814:13, please use with caution
"""
    magma_get_zpotrf_vbatched_crossover()

 Aux. vbatched routines
### Prototype
```c
magma_int_t magma_get_zpotrf_vbatched_crossover();
```
"""
function magma_get_zpotrf_vbatched_crossover()
    ccall((:magma_get_zpotrf_vbatched_crossover, libmagma), magma_int_t, ())
end

"""
    magma_getrf_vbatched_setup(m, n, stats, batchCount, queue)

getrf vbatched setup
### Prototype
```c
void magma_getrf_vbatched_setup( magma_int_t *m, magma_int_t *n, magma_int_t *stats, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_getrf_vbatched_setup(m, n, stats, batchCount, queue)
    ccall((:magma_getrf_vbatched_setup, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, stats, batchCount, queue)
end

"""
    setup_pivinfo_vbatched(pivinfo_array, pivinfo_offset, ipiv_array, ipiv_offset, m, n, max_m, nb, batchCount, queue)

getrf vbatched: setup pivinfo
### Prototype
```c
void setup_pivinfo_vbatched( magma_int_t **pivinfo_array, magma_int_t pivinfo_offset, magma_int_t **ipiv_array, magma_int_t ipiv_offset, magma_int_t* m, magma_int_t* n, magma_int_t max_m, magma_int_t nb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function setup_pivinfo_vbatched(pivinfo_array, pivinfo_offset, ipiv_array, ipiv_offset, m, n, max_m, nb, batchCount, queue)
    ccall((:setup_pivinfo_vbatched, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), pivinfo_array, pivinfo_offset, ipiv_array, ipiv_offset, m, n, max_m, nb, batchCount, queue)
end

"""
    adjust_ipiv_vbatched(ipiv_array, ipiv_offset, minmn, max_minmn, offset, batchCount, queue)

adjust pivot for LU
### Prototype
```c
void adjust_ipiv_vbatched( magma_int_t **ipiv_array, magma_int_t ipiv_offset, magma_int_t *minmn, magma_int_t max_minmn, magma_int_t offset, magma_int_t batchCount, magma_queue_t queue);
```
"""
function adjust_ipiv_vbatched(ipiv_array, ipiv_offset, minmn, max_minmn, offset, batchCount, queue)
    ccall((:adjust_ipiv_vbatched, libmagma), Cvoid, (Ptr{Ptr{magma_int_t}}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_int_t, magma_int_t, magma_queue_t), ipiv_array, ipiv_offset, minmn, max_minmn, offset, batchCount, queue)
end

"""
    magma_getrf_vbatched_checker(m, n, ldda, errors, batchCount, queue)

checker routines - LAPACK
### Prototype
```c
magma_int_t magma_getrf_vbatched_checker( magma_int_t* m, magma_int_t* n, magma_int_t* ldda, magma_int_t* errors, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_getrf_vbatched_checker(m, n, ldda, errors, batchCount, queue)
    ccall((:magma_getrf_vbatched_checker, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, ldda, errors, batchCount, queue)
end

"""
    magma_potrf_vbatched_checker(uplo, n, ldda, batchCount, queue)


### Prototype
```c
magma_int_t magma_potrf_vbatched_checker( magma_uplo_t uplo, magma_int_t* n, magma_int_t* ldda, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_potrf_vbatched_checker(uplo, n, ldda, batchCount, queue)
    ccall((:magma_potrf_vbatched_checker, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, ldda, batchCount, queue)
end

"""
    magma_gemm_vbatched_checker(transA, transB, m, n, k, ldda, lddb, lddc, batchCount, queue)

checker routines - Level 3 BLAS
### Prototype
```c
magma_int_t magma_gemm_vbatched_checker( magma_trans_t transA, magma_trans_t transB, magma_int_t* m, magma_int_t* n, magma_int_t* k, magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_gemm_vbatched_checker(transA, transB, m, n, k, ldda, lddb, lddc, batchCount, queue)
    ccall((:magma_gemm_vbatched_checker, libmagma), magma_int_t, (magma_trans_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), transA, transB, m, n, k, ldda, lddb, lddc, batchCount, queue)
end

"""
    magma_trsm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue)


### Prototype
```c
magma_int_t magma_trsm_vbatched_checker( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magma_int_t* ldda, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_trsm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue)
    ccall((:magma_trsm_vbatched_checker, libmagma), magma_int_t, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue)
end

"""
    magma_syrk_vbatched_checker(icomplex, uplo, trans, n, k, ldda, lddc, batchCount, queue)


### Prototype
```c
magma_int_t magma_syrk_vbatched_checker( magma_int_t icomplex, magma_uplo_t uplo, magma_trans_t trans, magma_int_t *n, magma_int_t *k, magma_int_t *ldda, magma_int_t *lddc, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_syrk_vbatched_checker(icomplex, uplo, trans, n, k, ldda, lddc, batchCount, queue)
    ccall((:magma_syrk_vbatched_checker, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), icomplex, uplo, trans, n, k, ldda, lddc, batchCount, queue)
end

"""
    magma_herk_vbatched_checker(uplo, trans, n, k, ldda, lddc, batchCount, queue)


### Prototype
```c
magma_int_t magma_herk_vbatched_checker( magma_uplo_t uplo, magma_trans_t trans, magma_int_t *n, magma_int_t *k, magma_int_t *ldda, magma_int_t *lddc, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_herk_vbatched_checker(uplo, trans, n, k, ldda, lddc, batchCount, queue)
    ccall((:magma_herk_vbatched_checker, libmagma), magma_int_t, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, ldda, lddc, batchCount, queue)
end

"""
    magma_syr2k_vbatched_checker(icomplex, uplo, trans, n, k, ldda, lddb, lddc, batchCount, queue)


### Prototype
```c
magma_int_t magma_syr2k_vbatched_checker( magma_int_t icomplex, magma_uplo_t uplo, magma_trans_t trans, magma_int_t *n, magma_int_t *k, magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_syr2k_vbatched_checker(icomplex, uplo, trans, n, k, ldda, lddb, lddc, batchCount, queue)
    ccall((:magma_syr2k_vbatched_checker, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), icomplex, uplo, trans, n, k, ldda, lddb, lddc, batchCount, queue)
end

"""
    magma_her2k_vbatched_checker(uplo, trans, n, k, ldda, lddb, lddc, batchCount, queue)


### Prototype
```c
magma_int_t magma_her2k_vbatched_checker( magma_uplo_t uplo, magma_trans_t trans, magma_int_t *n, magma_int_t *k, magma_int_t *ldda, magma_int_t *lddb, magma_int_t *lddc, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_her2k_vbatched_checker(uplo, trans, n, k, ldda, lddb, lddc, batchCount, queue)
    ccall((:magma_her2k_vbatched_checker, libmagma), magma_int_t, (magma_uplo_t, magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, trans, n, k, ldda, lddb, lddc, batchCount, queue)
end

"""
    magma_trmm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue)


### Prototype
```c
magma_int_t magma_trmm_vbatched_checker( magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t* m, magma_int_t* n, magma_int_t* ldda, magma_int_t* lddb, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_trmm_vbatched_checker(side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue)
    ccall((:magma_trmm_vbatched_checker, libmagma), magma_int_t, (magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, transA, diag, m, n, ldda, lddb, batchCount, queue)
end

"""
    magma_hemm_vbatched_checker(side, uplo, m, n, ldda, lddb, lddc, batchCount, queue)


### Prototype
```c
magma_int_t magma_hemm_vbatched_checker( magma_side_t side, magma_uplo_t uplo, magma_int_t* m, magma_int_t* n, magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_hemm_vbatched_checker(side, uplo, m, n, ldda, lddb, lddc, batchCount, queue)
    ccall((:magma_hemm_vbatched_checker, libmagma), magma_int_t, (magma_side_t, magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), side, uplo, m, n, ldda, lddb, lddc, batchCount, queue)
end

"""
    magma_gemv_vbatched_checker(trans, m, n, ldda, incx, incy, batchCount, queue)

checker routines - Level 2 BLAS
### Prototype
```c
magma_int_t magma_gemv_vbatched_checker( magma_trans_t trans, magma_int_t* m, magma_int_t* n, magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_gemv_vbatched_checker(trans, m, n, ldda, incx, incy, batchCount, queue)
    ccall((:magma_gemv_vbatched_checker, libmagma), magma_int_t, (magma_trans_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), trans, m, n, ldda, incx, incy, batchCount, queue)
end

"""
    magma_hemv_vbatched_checker(uplo, n, ldda, incx, incy, batchCount, queue)


### Prototype
```c
magma_int_t magma_hemv_vbatched_checker( magma_uplo_t uplo, magma_int_t* n, magma_int_t* ldda, magma_int_t* incx, magma_int_t* incy, magma_int_t batchCount, magma_queue_t queue );
```
"""
function magma_hemv_vbatched_checker(uplo, n, ldda, incx, incy, batchCount, queue)
    ccall((:magma_hemv_vbatched_checker, libmagma), magma_int_t, (magma_uplo_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), uplo, n, ldda, incx, incy, batchCount, queue)
end

"""
    magma_axpy_vbatched_checker(n, incx, incy, batchCount, queue)

checker routines - Level 1 BLAS
### Prototype
```c
magma_int_t magma_axpy_vbatched_checker( magma_int_t *n, magma_int_t *incx, magma_int_t *incy, magma_int_t batchCount, magma_queue_t queue);
```
"""
function magma_axpy_vbatched_checker(n, incx, incy, batchCount, queue)
    ccall((:magma_axpy_vbatched_checker, libmagma), magma_int_t, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, incx, incy, batchCount, queue)
end

"""
    magma_imax_size_1(n, l, queue)

routines to find the maximum dimensions
### Prototype
```c
void magma_imax_size_1(magma_int_t *n, magma_int_t l, magma_queue_t queue);
```
"""
function magma_imax_size_1(n, l, queue)
    ccall((:magma_imax_size_1, libmagma), Cvoid, (Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, l, queue)
end

"""
    magma_imax_size_2(m, n, l, queue)


### Prototype
```c
void magma_imax_size_2(magma_int_t *m, magma_int_t *n, magma_int_t l, magma_queue_t queue);
```
"""
function magma_imax_size_2(m, n, l, queue)
    ccall((:magma_imax_size_2, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, l, queue)
end

"""
    magma_imax_size_3(m, n, k, l, queue)


### Prototype
```c
void magma_imax_size_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, magma_int_t l, magma_queue_t queue);
```
"""
function magma_imax_size_3(m, n, k, l, queue)
    ccall((:magma_imax_size_3, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), m, n, k, l, queue)
end

"""
    magma_ivec_max(vecsize, x, work, lwork, queue)

aux. routines
### Prototype
```c
magma_int_t magma_ivec_max( magma_int_t vecsize, magma_int_t* x, magma_int_t* work, magma_int_t lwork, magma_queue_t queue);
```
"""
function magma_ivec_max(vecsize, x, work, lwork, queue)
    ccall((:magma_ivec_max, libmagma), magma_int_t, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), vecsize, x, work, lwork, queue)
end

"""
    magma_isum_reduce(vecsize, x, work, lwork, queue)


### Prototype
```c
magma_int_t magma_isum_reduce( magma_int_t vecsize, magma_int_t* x, magma_int_t* work, magma_int_t lwork, magma_queue_t queue);
```
"""
function magma_isum_reduce(vecsize, x, work, lwork, queue)
    ccall((:magma_isum_reduce, libmagma), magma_int_t, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), vecsize, x, work, lwork, queue)
end

"""
    magma_ivec_add(vecsize, a1, x1, a2, x2, y, queue)


### Prototype
```c
void magma_ivec_add( magma_int_t vecsize, magma_int_t a1, magma_int_t *x1, magma_int_t a2, magma_int_t *x2, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_add(vecsize, a1, x1, a2, x2, y, queue)
    ccall((:magma_ivec_add, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_queue_t), vecsize, a1, x1, a2, x2, y, queue)
end

"""
    magma_ivec_mul(vecsize, x1, x2, y, queue)


### Prototype
```c
void magma_ivec_mul( magma_int_t vecsize, magma_int_t *x1, magma_int_t *x2, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_mul(vecsize, x1, x2, y, queue)
    ccall((:magma_ivec_mul, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_queue_t), vecsize, x1, x2, y, queue)
end

"""
    magma_ivec_ceildiv(vecsize, x, nb, y, queue)


### Prototype
```c
void magma_ivec_ceildiv( magma_int_t vecsize, magma_int_t *x, magma_int_t nb, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_ceildiv(vecsize, x, nb, y, queue)
    ccall((:magma_ivec_ceildiv, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, x, nb, y, queue)
end

"""
    magma_ivec_roundup(vecsize, x, nb, y, queue)


### Prototype
```c
void magma_ivec_roundup( magma_int_t vecsize, magma_int_t *x, magma_int_t nb, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_roundup(vecsize, x, nb, y, queue)
    ccall((:magma_ivec_roundup, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, x, nb, y, queue)
end

"""
    magma_ivec_setc(vecsize, x, value, queue)


### Prototype
```c
void magma_ivec_setc( magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_queue_t queue);
```
"""
function magma_ivec_setc(vecsize, x, value, queue)
    ccall((:magma_ivec_setc, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), vecsize, x, value, queue)
end

"""
    magma_zsetvector_const(vecsize, x, value, queue)


### Prototype
```c
void magma_zsetvector_const( magma_int_t vecsize, magmaDoubleComplex *x, magmaDoubleComplex value, magma_queue_t queue);
```
"""
function magma_zsetvector_const(vecsize, x, value, queue)
    ccall((:magma_zsetvector_const, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex, magma_queue_t), vecsize, x, value, queue)
end

"""
    magma_csetvector_const(vecsize, x, value, queue)


### Prototype
```c
void magma_csetvector_const( magma_int_t vecsize, magmaFloatComplex *x, magmaFloatComplex value, magma_queue_t queue);
```
"""
function magma_csetvector_const(vecsize, x, value, queue)
    ccall((:magma_csetvector_const, libmagma), Cvoid, (magma_int_t, Ptr{magmaFloatComplex}, magmaFloatComplex, magma_queue_t), vecsize, x, value, queue)
end

"""
    magma_dsetvector_const(vecsize, x, value, queue)


### Prototype
```c
void magma_dsetvector_const( magma_int_t vecsize, double *x, double value, magma_queue_t queue);
```
"""
function magma_dsetvector_const(vecsize, x, value, queue)
    ccall((:magma_dsetvector_const, libmagma), Cvoid, (magma_int_t, Ptr{Cdouble}, Cdouble, magma_queue_t), vecsize, x, value, queue)
end

"""
    magma_ssetvector_const(vecsize, x, value, queue)


### Prototype
```c
void magma_ssetvector_const( magma_int_t vecsize, float *x, float value, magma_queue_t queue);
```
"""
function magma_ssetvector_const(vecsize, x, value, queue)
    ccall((:magma_ssetvector_const, libmagma), Cvoid, (magma_int_t, Ptr{Cfloat}, Cfloat, magma_queue_t), vecsize, x, value, queue)
end

"""
    magma_ivec_addc(vecsize, x, value, y, queue)


### Prototype
```c
void magma_ivec_addc( magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_addc(vecsize, x, value, y, queue)
    ccall((:magma_ivec_addc, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, x, value, y, queue)
end

"""
    magma_ivec_mulc(vecsize, x, value, y, queue)


### Prototype
```c
void magma_ivec_mulc( magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_mulc(vecsize, x, value, y, queue)
    ccall((:magma_ivec_mulc, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, x, value, y, queue)
end

"""
    magma_ivec_minc(vecsize, x, value, y, queue)


### Prototype
```c
void magma_ivec_minc( magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_minc(vecsize, x, value, y, queue)
    ccall((:magma_ivec_minc, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, x, value, y, queue)
end

"""
    magma_ivec_maxc(vecsize, x, value, y, queue)


### Prototype
```c
void magma_ivec_maxc( magma_int_t vecsize, magma_int_t* x, magma_int_t value, magma_int_t* y, magma_queue_t queue);
```
"""
function magma_ivec_maxc(vecsize, x, value, y, queue)
    ccall((:magma_ivec_maxc, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, x, value, y, queue)
end

"""
    magma_ivec_min_vv(vecsize, v1, v2, y, queue)


### Prototype
```c
void magma_ivec_min_vv( magma_int_t vecsize, magma_int_t *v1, magma_int_t *v2, magma_int_t *y, magma_queue_t queue);
```
"""
function magma_ivec_min_vv(vecsize, v1, v2, y, queue)
    ccall((:magma_ivec_min_vv, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_queue_t), vecsize, v1, v2, y, queue)
end

"""
    magma_compute_trsm_jb(vecsize, m, tri_nb, jbv, queue)


### Prototype
```c
void magma_compute_trsm_jb( magma_int_t vecsize, magma_int_t* m, magma_int_t tri_nb, magma_int_t* jbv, magma_queue_t queue);
```
"""
function magma_compute_trsm_jb(vecsize, m, tri_nb, jbv, queue)
    ccall((:magma_compute_trsm_jb, libmagma), Cvoid, (magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_queue_t), vecsize, m, tri_nb, jbv, queue)
end

"""
    magma_prefix_sum_inplace(ivec, length, queue)


### Prototype
```c
void magma_prefix_sum_inplace(magma_int_t* ivec, magma_int_t length, magma_queue_t queue);
```
"""
function magma_prefix_sum_inplace(ivec, length, queue)
    ccall((:magma_prefix_sum_inplace, libmagma), Cvoid, (Ptr{magma_int_t}, magma_int_t, magma_queue_t), ivec, length, queue)
end

"""
    magma_prefix_sum_outofplace(ivec, ovec, length, queue)


### Prototype
```c
void magma_prefix_sum_outofplace(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_queue_t queue);
```
"""
function magma_prefix_sum_outofplace(ivec, ovec, length, queue)
    ccall((:magma_prefix_sum_outofplace, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t), ivec, ovec, length, queue)
end

"""
    magma_prefix_sum_inplace_w(ivec, length, workspace, lwork, queue)


### Prototype
```c
void magma_prefix_sum_inplace_w(magma_int_t* ivec, magma_int_t length, magma_int_t* workspace, magma_int_t lwork, magma_queue_t queue);
```
"""
function magma_prefix_sum_inplace_w(ivec, length, workspace, lwork, queue)
    ccall((:magma_prefix_sum_inplace_w, libmagma), Cvoid, (Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), ivec, length, workspace, lwork, queue)
end

"""
    magma_prefix_sum_outofplace_w(ivec, ovec, length, workspace, lwork, queue)


### Prototype
```c
void magma_prefix_sum_outofplace_w(magma_int_t* ivec, magma_int_t* ovec, magma_int_t length, magma_int_t* workspace, magma_int_t lwork, magma_queue_t queue);
```
"""
function magma_prefix_sum_outofplace_w(ivec, ovec, length, workspace, lwork, queue)
    ccall((:magma_prefix_sum_outofplace_w, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), ivec, ovec, length, workspace, lwork, queue)
end

"""
    magma_zbulge_applyQ_v2(side, NE, n, nb, Vblksiz, dE, ldde, V, ldv, T, ldt, info)


### Prototype
```c
magma_int_t magma_zbulge_applyQ_v2( magma_side_t side, magma_int_t NE, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magmaDoubleComplex_ptr dE, magma_int_t ldde, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *T, magma_int_t ldt, magma_int_t *info);
```
"""
function magma_zbulge_applyQ_v2(side, NE, n, nb, Vblksiz, dE, ldde, V, ldv, T, ldt, info)
    ccall((:magma_zbulge_applyQ_v2, libmagma), magma_int_t, (magma_side_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, NE, n, nb, Vblksiz, dE, ldde, V, ldv, T, ldt, info)
end

"""
    magma_zbulge_applyQ_v2_m(ngpu, side, NE, n, nb, Vblksiz, E, lde, V, ldv, T, ldt, info)


### Prototype
```c
magma_int_t magma_zbulge_applyQ_v2_m( magma_int_t ngpu, magma_side_t side, magma_int_t NE, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magmaDoubleComplex *E, magma_int_t lde, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *T, magma_int_t ldt, magma_int_t *info);
```
"""
function magma_zbulge_applyQ_v2_m(ngpu, side, NE, n, nb, Vblksiz, E, lde, V, ldv, T, ldt, info)
    ccall((:magma_zbulge_applyQ_v2_m, libmagma), magma_int_t, (magma_int_t, magma_side_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, side, NE, n, nb, Vblksiz, E, lde, V, ldv, T, ldt, info)
end

"""
    magma_zbulge_back(uplo, n, nb, ne, Vblksiz, Z, ldz, dZ, lddz, V, ldv, TAU, T, ldt, info)


### Prototype
```c
magma_int_t magma_zbulge_back( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t ne, magma_int_t Vblksiz, magmaDoubleComplex *Z, magma_int_t ldz, magmaDoubleComplex_ptr dZ, magma_int_t lddz, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t ldt, magma_int_t* info);
```
"""
function magma_zbulge_back(uplo, n, nb, ne, Vblksiz, Z, ldz, dZ, lddz, V, ldv, TAU, T, ldt, info)
    ccall((:magma_zbulge_back, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, nb, ne, Vblksiz, Z, ldz, dZ, lddz, V, ldv, TAU, T, ldt, info)
end

"""
    magma_zbulge_back_m(ngpu, uplo, n, nb, ne, Vblksiz, Z, ldz, V, ldv, TAU, T, ldt, info)


### Prototype
```c
magma_int_t magma_zbulge_back_m( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t ne, magma_int_t Vblksiz, magmaDoubleComplex *Z, magma_int_t ldz, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t ldt, magma_int_t* info);
```
"""
function magma_zbulge_back_m(ngpu, uplo, n, nb, ne, Vblksiz, Z, ldz, V, ldv, TAU, T, ldt, info)
    ccall((:magma_zbulge_back_m, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, uplo, n, nb, ne, Vblksiz, Z, ldz, V, ldv, TAU, T, ldt, info)
end

"""
    magma_ztrdtype1cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)


### Prototype
```c
void magma_ztrdtype1cbHLsym_withQ_v2( magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, magmaDoubleComplex *work);
```
"""
function magma_ztrdtype1cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)
    ccall((:magma_ztrdtype1cbHLsym_withQ_v2, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}), n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)
end

"""
    magma_ztrdtype2cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)


### Prototype
```c
void magma_ztrdtype2cbHLsym_withQ_v2( magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, magmaDoubleComplex *work);
```
"""
function magma_ztrdtype2cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)
    ccall((:magma_ztrdtype2cbHLsym_withQ_v2, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}), n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)
end

"""
    magma_ztrdtype3cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)


### Prototype
```c
void magma_ztrdtype3cbHLsym_withQ_v2( magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, magmaDoubleComplex *work);
```
"""
function magma_ztrdtype3cbHLsym_withQ_v2(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)
    ccall((:magma_ztrdtype3cbHLsym_withQ_v2, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}), n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, work)
end

"""
    magma_zlarfy(n, A, lda, V, TAU, work)


### Prototype
```c
void magma_zlarfy( magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, const magmaDoubleComplex *V, const magmaDoubleComplex *TAU, magmaDoubleComplex *work);
```
"""
function magma_zlarfy(n, A, lda, V, TAU, work)
    ccall((:magma_zlarfy, libmagma), Cvoid, (magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}), n, A, lda, V, TAU, work)
end

"""
    magma_zhbtype1cb(n, nb, A, lda, V, LDV, TAU, st, ed, sweep, Vblksiz, wantz, work)


### Prototype
```c
void magma_zhbtype1cb(magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *V, magma_int_t LDV, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, magma_int_t wantz, magmaDoubleComplex *work);
```
"""
function magma_zhbtype1cb(n, nb, A, lda, V, LDV, TAU, st, ed, sweep, Vblksiz, wantz, work)
    ccall((:magma_zhbtype1cb, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}), n, nb, A, lda, V, LDV, TAU, st, ed, sweep, Vblksiz, wantz, work)
end

"""
    magma_zhbtype2cb(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, wantz, work)


### Prototype
```c
void magma_zhbtype2cb(magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, magma_int_t wantz, magmaDoubleComplex *work);
```
"""
function magma_zhbtype2cb(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, wantz, work)
    ccall((:magma_zhbtype2cb, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}), n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, wantz, work)
end

"""
    magma_zhbtype3cb(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, wantz, work)


### Prototype
```c
void magma_zhbtype3cb(magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t st, magma_int_t ed, magma_int_t sweep, magma_int_t Vblksiz, magma_int_t wantz, magmaDoubleComplex *work);
```
"""
function magma_zhbtype3cb(n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, wantz, work)
    ccall((:magma_zhbtype3cb, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}), n, nb, A, lda, V, ldv, TAU, st, ed, sweep, Vblksiz, wantz, work)
end

"""
    magma_zunmqr_2stage_gpu(side, trans, m, n, k, dA, ldda, dC, lddc, dT, nb, info)


### Prototype
```c
magma_int_t magma_zunmqr_2stage_gpu( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dC, magma_int_t lddc, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zunmqr_2stage_gpu(side, trans, m, n, k, dA, ldda, dC, lddc, dT, nb, info)
    ccall((:magma_zunmqr_2stage_gpu, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, dA, ldda, dC, lddc, dT, nb, info)
end

"""
    magma_get_zbulge_lq2(n, threads, wantz)


### Prototype
```c
magma_int_t magma_get_zbulge_lq2( magma_int_t n, magma_int_t threads, magma_int_t wantz);
```
"""
function magma_get_zbulge_lq2(n, threads, wantz)
    ccall((:magma_get_zbulge_lq2, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t), n, threads, wantz)
end

"""
    magma_zbulge_getstg2size(n, nb, wantz, Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2)


### Prototype
```c
magma_int_t magma_zbulge_getstg2size(magma_int_t n, magma_int_t nb, magma_int_t wantz, magma_int_t Vblksiz, magma_int_t ldv, magma_int_t ldt, magma_int_t *blkcnt, magma_int_t *sizTAU2, magma_int_t *sizT2, magma_int_t *sizV2);
```
"""
function magma_zbulge_getstg2size(n, nb, wantz, Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2)
    ccall((:magma_zbulge_getstg2size, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, wantz, Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2)
end

"""
    magma_zbulge_getlwstg2(n, threads, wantz, Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2)


### Prototype
```c
magma_int_t magma_zbulge_getlwstg2(magma_int_t n, magma_int_t threads, magma_int_t wantz, magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt, magma_int_t *blkcnt, magma_int_t *sizTAU2, magma_int_t *sizT2, magma_int_t *sizV2);
```
"""
function magma_zbulge_getlwstg2(n, threads, wantz, Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2)
    ccall((:magma_zbulge_getlwstg2, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), n, threads, wantz, Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2)
end

"""
    magma_bulge_get_VTsiz(n, nb, threads, Vblksiz, ldv, ldt)


### Prototype
```c
void magma_bulge_get_VTsiz(magma_int_t n, magma_int_t nb, magma_int_t threads, magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt);
```
"""
function magma_bulge_get_VTsiz(n, nb, threads, Vblksiz, ldv, ldt)
    ccall((:magma_bulge_get_VTsiz, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, threads, Vblksiz, ldv, ldt)
end

"""
    magma_zheevdx_getworksize(n, threads, wantz, lwmin, lrwmin, liwmin)


### Prototype
```c
void magma_zheevdx_getworksize(magma_int_t n, magma_int_t threads, magma_int_t wantz, magma_int_t *lwmin, #ifdef MAGMA_COMPLEX magma_int_t *lrwmin, #endif magma_int_t *liwmin);
```
"""
function magma_zheevdx_getworksize(n, threads, wantz, lwmin, lrwmin, liwmin)
    ccall((:magma_zheevdx_getworksize, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), n, threads, wantz, lwmin, lrwmin, liwmin)
end

"""
    magma_zhetrd_bhe2trc_v5(threads, wantz, uplo, ne, n, nb, A, lda, D, E, dT1, ldt1)

used only for old version and internal
### Prototype
```c
magma_int_t magma_zhetrd_bhe2trc_v5( magma_int_t threads, magma_int_t wantz, magma_uplo_t uplo, magma_int_t ne, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, double *D, double *E, magmaDoubleComplex_ptr dT1, magma_int_t ldt1);
```
"""
function magma_zhetrd_bhe2trc_v5(threads, wantz, uplo, ne, n, nb, A, lda, D, E, dT1, ldt1)
    ccall((:magma_zhetrd_bhe2trc_v5, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, magmaDoubleComplex_ptr, magma_int_t), threads, wantz, uplo, ne, n, nb, A, lda, D, E, dT1, ldt1)
end

"""
    magma_zungqr_2stage_gpu(m, n, k, dA, ldda, tau, dT, nb, info)


### Prototype
```c
magma_int_t magma_zungqr_2stage_gpu( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zungqr_2stage_gpu(m, n, k, dA, ldda, tau, dT, nb, info)
    ccall((:magma_zungqr_2stage_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, k, dA, ldda, tau, dT, nb, info)
end

# no prototype is found for this function at magma_bulge.h:22:17, please use with caution
"""
    magma_yield()


### Prototype
```c
magma_int_t magma_yield();
```
"""
function magma_yield()
    ccall((:magma_yield, libmagma), magma_int_t, ())
end

"""
    magma_bulge_getlwstg1(n, nb, lda2)


### Prototype
```c
magma_int_t magma_bulge_getlwstg1(magma_int_t n, magma_int_t nb, magma_int_t *lda2);
```
"""
function magma_bulge_getlwstg1(n, nb, lda2)
    ccall((:magma_bulge_getlwstg1, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magma_int_t}), n, nb, lda2)
end

"""
    cmp_vals(n, wr1, wr2, nrmI, nrm1, nrm2)


### Prototype
```c
void cmp_vals(magma_int_t n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);
```
"""
function cmp_vals(n, wr1, wr2, nrmI, nrm1, nrm2)
    ccall((:cmp_vals, libmagma), Cvoid, (magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), n, wr1, wr2, nrmI, nrm1, nrm2)
end

"""
    magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep, st, ldv, Vpos, TAUpos)


### Prototype
```c
void magma_bulge_findVTAUpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t *Vpos, magma_int_t *TAUpos);
```
"""
function magma_bulge_findVTAUpos(n, nb, Vblksiz, sweep, st, ldv, Vpos, TAUpos)
    ccall((:magma_bulge_findVTAUpos, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, Vblksiz, sweep, st, ldv, Vpos, TAUpos)
end

"""
    magma_bulge_findVTpos(n, nb, Vblksiz, sweep, st, ldv, ldt, Vpos, Tpos)


### Prototype
```c
void magma_bulge_findVTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t ldt, magma_int_t *Vpos, magma_int_t *Tpos);
```
"""
function magma_bulge_findVTpos(n, nb, Vblksiz, sweep, st, ldv, ldt, Vpos, Tpos)
    ccall((:magma_bulge_findVTpos, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, Vblksiz, sweep, st, ldv, ldt, Vpos, Tpos)
end

"""
    magma_bulge_findVTAUTpos(n, nb, Vblksiz, sweep, st, ldv, ldt, Vpos, TAUpos, Tpos, blkid)


### Prototype
```c
void magma_bulge_findVTAUTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t ldv, magma_int_t ldt, magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *blkid);
```
"""
function magma_bulge_findVTAUTpos(n, nb, Vblksiz, sweep, st, ldv, ldt, Vpos, TAUpos, Tpos, blkid)
    ccall((:magma_bulge_findVTAUTpos, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, Vblksiz, sweep, st, ldv, ldt, Vpos, TAUpos, Tpos, blkid)
end

"""
    magma_bulge_findpos(n, nb, Vblksiz, sweep, st, myblkid)


### Prototype
```c
void magma_bulge_findpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *myblkid);
```
"""
function magma_bulge_findpos(n, nb, Vblksiz, sweep, st, myblkid)
    ccall((:magma_bulge_findpos, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}), n, nb, Vblksiz, sweep, st, myblkid)
end

"""
    magma_bulge_findpos113(n, nb, Vblksiz, sweep, st, myblkid)


### Prototype
```c
void magma_bulge_findpos113(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *myblkid);
```
"""
function magma_bulge_findpos113(n, nb, Vblksiz, sweep, st, myblkid)
    ccall((:magma_bulge_findpos113, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}), n, nb, Vblksiz, sweep, st, myblkid)
end

"""
    magma_bulge_get_blkcnt(n, nb, Vblksiz)


### Prototype
```c
magma_int_t magma_bulge_get_blkcnt(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz);
```
"""
function magma_bulge_get_blkcnt(n, nb, Vblksiz)
    ccall((:magma_bulge_get_blkcnt, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t), n, nb, Vblksiz)
end

"""
    findVTpos(n, nb, Vblksiz, sweep, st, Vpos, TAUpos, Tpos, myblkid)


### Prototype
```c
void findVTpos(magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magma_int_t sweep, magma_int_t st, magma_int_t *Vpos, magma_int_t *TAUpos, magma_int_t *Tpos, magma_int_t *myblkid);
```
"""
function findVTpos(n, nb, Vblksiz, sweep, st, Vpos, TAUpos, Tpos, myblkid)
    ccall((:findVTpos, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), n, nb, Vblksiz, sweep, st, Vpos, TAUpos, Tpos, myblkid)
end

"""
    zgehrd_data

//**
Structure containing matrices for multi-GPU zgehrd.

- dA  is distributed column block-cyclic across GPUs.
- dV  is duplicated on all GPUs.
- dVd is distributed row block-cyclic across GPUs (TODO: verify).
- dY  is partial results on each GPU in zlahr2,
then complete results are duplicated on all GPUs for zlahru.
- dW  is local to each GPU (workspace).
- dTi is duplicated on all GPUs.

@ingroup magma_gehrd
"""
struct zgehrd_data
    ngpu::magma_int_t
    ldda::magma_int_t
    ldv::magma_int_t
    ldvd::magma_int_t
    dA::NTuple{8, magmaDoubleComplex_ptr}
    dV::NTuple{8, magmaDoubleComplex_ptr}
    dVd::NTuple{8, magmaDoubleComplex_ptr}
    dY::NTuple{8, magmaDoubleComplex_ptr}
    dW::NTuple{8, magmaDoubleComplex_ptr}
    dTi::NTuple{8, magmaDoubleComplex_ptr}
    queues::NTuple{8, magma_queue_t}
end

"""
    magma_get_zpotrf_nb(n)

Cholesky, LU, symmetric indefinite
### Prototype
```c
magma_int_t magma_get_zpotrf_nb( magma_int_t n );
```
"""
function magma_get_zpotrf_nb(n)
    ccall((:magma_get_zpotrf_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zgetrf_nb(m, n)


### Prototype
```c
magma_int_t magma_get_zgetrf_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgetrf_nb(m, n)
    ccall((:magma_get_zgetrf_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgetrf_native_nb(m, n)


### Prototype
```c
magma_int_t magma_get_zgetrf_native_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgetrf_native_nb(m, n)
    ccall((:magma_get_zgetrf_native_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgetri_nb(n)


### Prototype
```c
magma_int_t magma_get_zgetri_nb( magma_int_t n );
```
"""
function magma_get_zgetri_nb(n)
    ccall((:magma_get_zgetri_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zhetrf_nb(n)


### Prototype
```c
magma_int_t magma_get_zhetrf_nb( magma_int_t n );
```
"""
function magma_get_zhetrf_nb(n)
    ccall((:magma_get_zhetrf_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zhetrf_nopiv_nb(n)


### Prototype
```c
magma_int_t magma_get_zhetrf_nopiv_nb( magma_int_t n );
```
"""
function magma_get_zhetrf_nopiv_nb(n)
    ccall((:magma_get_zhetrf_nopiv_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zhetrf_aasen_nb(n)


### Prototype
```c
magma_int_t magma_get_zhetrf_aasen_nb( magma_int_t n );
```
"""
function magma_get_zhetrf_aasen_nb(n)
    ccall((:magma_get_zhetrf_aasen_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zgeqp3_nb(m, n)

QR
### Prototype
```c
magma_int_t magma_get_zgeqp3_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgeqp3_nb(m, n)
    ccall((:magma_get_zgeqp3_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgeqrf_nb(m, n)


### Prototype
```c
magma_int_t magma_get_zgeqrf_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgeqrf_nb(m, n)
    ccall((:magma_get_zgeqrf_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgeqlf_nb(m, n)


### Prototype
```c
magma_int_t magma_get_zgeqlf_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgeqlf_nb(m, n)
    ccall((:magma_get_zgeqlf_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgelqf_nb(m, n)


### Prototype
```c
magma_int_t magma_get_zgelqf_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgelqf_nb(m, n)
    ccall((:magma_get_zgelqf_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgehrd_nb(n)

eigenvalues
### Prototype
```c
magma_int_t magma_get_zgehrd_nb( magma_int_t n );
```
"""
function magma_get_zgehrd_nb(n)
    ccall((:magma_get_zgehrd_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zhetrd_nb(n)


### Prototype
```c
magma_int_t magma_get_zhetrd_nb( magma_int_t n );
```
"""
function magma_get_zhetrd_nb(n)
    ccall((:magma_get_zhetrd_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zhegst_nb(n)


### Prototype
```c
magma_int_t magma_get_zhegst_nb( magma_int_t n );
```
"""
function magma_get_zhegst_nb(n)
    ccall((:magma_get_zhegst_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zhegst_m_nb(n)


### Prototype
```c
magma_int_t magma_get_zhegst_m_nb( magma_int_t n );
```
"""
function magma_get_zhegst_m_nb(n)
    ccall((:magma_get_zhegst_m_nb, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zgebrd_nb(m, n)

SVD
### Prototype
```c
magma_int_t magma_get_zgebrd_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgebrd_nb(m, n)
    ccall((:magma_get_zgebrd_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zgesvd_nb(m, n)


### Prototype
```c
magma_int_t magma_get_zgesvd_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_zgesvd_nb(m, n)
    ccall((:magma_get_zgesvd_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_zbulge_nb(n, nbthreads)

2-stage eigenvalues
### Prototype
```c
magma_int_t magma_get_zbulge_nb( magma_int_t n, magma_int_t nbthreads );
```
"""
function magma_get_zbulge_nb(n, nbthreads)
    ccall((:magma_get_zbulge_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), n, nbthreads)
end

"""
    magma_get_zbulge_nb_mgpu(n)


### Prototype
```c
magma_int_t magma_get_zbulge_nb_mgpu( magma_int_t n );
```
"""
function magma_get_zbulge_nb_mgpu(n)
    ccall((:magma_get_zbulge_nb_mgpu, libmagma), magma_int_t, (magma_int_t,), n)
end

"""
    magma_get_zbulge_vblksiz(n, nb, nbthreads)


### Prototype
```c
magma_int_t magma_get_zbulge_vblksiz( magma_int_t n, magma_int_t nb, magma_int_t nbthreads );
```
"""
function magma_get_zbulge_vblksiz(n, nb, nbthreads)
    ccall((:magma_get_zbulge_vblksiz, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t), n, nb, nbthreads)
end

# no prototype is found for this function at magma_z.h:59:13, please use with caution
"""
    magma_get_zbulge_gcperf()


### Prototype
```c
magma_int_t magma_get_zbulge_gcperf();
```
"""
function magma_get_zbulge_gcperf()
    ccall((:magma_get_zbulge_gcperf, libmagma), magma_int_t, ())
end

"""
    magma_zgebrd(m, n, A, lda, d, e, tauq, taup, work, lwork, info)

------------------------------------------------------------ zge routines
### Prototype
```c
magma_int_t magma_zgebrd( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *d, double *e, magmaDoubleComplex *tauq, magmaDoubleComplex *taup, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgebrd(m, n, A, lda, d, e, tauq, taup, work, lwork, info)
    ccall((:magma_zgebrd, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, d, e, tauq, taup, work, lwork, info)
end

"""
    magma_zgeev(jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, work, lwork, rwork, info)


### Prototype
```c
magma_int_t magma_zgeev( magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, #ifdef MAGMA_COMPLEX magmaDoubleComplex *w, #else double *wr, double *wi, #endif magmaDoubleComplex *VL, magma_int_t ldvl, magmaDoubleComplex *VR, magma_int_t ldvr, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_zgeev(jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, work, lwork, rwork, info)
    ccall((:magma_zgeev, libmagma), magma_int_t, (magma_vec_t, magma_vec_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, work, lwork, rwork, info)
end

"""
    magma_zgeev_m(jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, work, lwork, rwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeev_m( magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, #ifdef MAGMA_COMPLEX magmaDoubleComplex *w, #else double *wr, double *wi, #endif magmaDoubleComplex *VL, magma_int_t ldvl, magmaDoubleComplex *VR, magma_int_t ldvr, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_zgeev_m(jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, work, lwork, rwork, info)
    ccall((:magma_zgeev_m, libmagma), magma_int_t, (magma_vec_t, magma_vec_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), jobvl, jobvr, n, A, lda, w, VL, ldvl, VR, ldvr, work, lwork, rwork, info)
end

"""
    magma_zgegqr_gpu(ikind, m, n, dA, ldda, dwork, work, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgegqr_gpu( magma_int_t ikind, magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dwork, magmaDoubleComplex *work, magma_int_t *info);
```
"""
function magma_zgegqr_gpu(ikind, m, n, dA, ldda, dwork, work, info)
    ccall((:magma_zgegqr_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}), ikind, m, n, dA, ldda, dwork, work, info)
end

"""
    magma_zgehrd(n, ilo, ihi, A, lda, tau, work, lwork, dT, info)


### Prototype
```c
magma_int_t magma_zgehrd( magma_int_t n, magma_int_t ilo, magma_int_t ihi, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex_ptr dT, magma_int_t *info);
```
"""
function magma_zgehrd(n, ilo, ihi, A, lda, tau, work, lwork, dT, info)
    ccall((:magma_zgehrd, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, Ptr{magma_int_t}), n, ilo, ihi, A, lda, tau, work, lwork, dT, info)
end

"""
    magma_zgehrd_m(n, ilo, ihi, A, lda, tau, work, lwork, T, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgehrd_m( magma_int_t n, magma_int_t ilo, magma_int_t ihi, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex *T, magma_int_t *info);
```
"""
function magma_zgehrd_m(n, ilo, ihi, A, lda, tau, work, lwork, T, info)
    ccall((:magma_zgehrd_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}), n, ilo, ihi, A, lda, tau, work, lwork, T, info)
end

"""
    magma_zgehrd2(n, ilo, ihi, A, lda, tau, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgehrd2( magma_int_t n, magma_int_t ilo, magma_int_t ihi, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgehrd2(n, ilo, ihi, A, lda, tau, work, lwork, info)
    ccall((:magma_zgehrd2, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), n, ilo, ihi, A, lda, tau, work, lwork, info)
end

"""
    magma_zgelqf(m, n, A, lda, tau, work, lwork, info)


### Prototype
```c
magma_int_t magma_zgelqf( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgelqf(m, n, A, lda, tau, work, lwork, info)
    ccall((:magma_zgelqf, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, tau, work, lwork, info)
end

"""
    magma_zgelqf_gpu(m, n, dA, ldda, tau, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgelqf_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgelqf_gpu(m, n, dA, ldda, tau, work, lwork, info)
    ccall((:magma_zgelqf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, dA, ldda, tau, work, lwork, info)
end

"""
    magma_zgels(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgels( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr A, magma_int_t lda, magmaDoubleComplex_ptr B, magma_int_t ldb, magmaDoubleComplex *hwork, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgels(trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork, info)
    ccall((:magma_zgels, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), trans, m, n, nrhs, A, lda, B, ldb, hwork, lwork, info)
end

"""
    magma_zggrqf(m, p, n, A, lda, taua, B, ldb, taub, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zggrqf( magma_int_t m, magma_int_t p, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *taua, magmaDoubleComplex *B, magma_int_t ldb, magmaDoubleComplex *taub, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zggrqf(m, p, n, A, lda, taua, B, ldb, taub, work, lwork, info)
    ccall((:magma_zggrqf, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, p, n, A, lda, taua, B, ldb, taub, work, lwork, info)
end

"""
    magma_zgglse(m, n, p, A, lda, B, ldb, c, d, x, work, lwork, info)


### Prototype
```c
magma_int_t magma_zgglse( magma_int_t m, magma_int_t n, magma_int_t p, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, magmaDoubleComplex *c, magmaDoubleComplex *d, magmaDoubleComplex *x, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgglse(m, n, p, A, lda, B, ldb, c, d, x, work, lwork, info)
    ccall((:magma_zgglse, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, p, A, lda, B, ldb, c, d, x, work, lwork, info)
end

"""
    magma_zgels_gpu(trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)


### Prototype
```c
magma_int_t magma_zgels_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex *hwork, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgels_gpu(trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)
    ccall((:magma_zgels_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)
end

"""
    magma_zgels3_gpu(trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgels3_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex *hwork, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgels3_gpu(trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)
    ccall((:magma_zgels3_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), trans, m, n, nrhs, dA, ldda, dB, lddb, hwork, lwork, info)
end

"""
    magma_zgeqlf(m, n, A, lda, tau, work, lwork, info)


### Prototype
```c
magma_int_t magma_zgeqlf( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgeqlf(m, n, A, lda, tau, work, lwork, info)
    ccall((:magma_zgeqlf, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, tau, work, lwork, info)
end

"""
    magma_zgeqp3(m, n, A, lda, jpvt, tau, work, lwork, rwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqp3( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *jpvt, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_zgeqp3(m, n, A, lda, jpvt, tau, work, lwork, rwork, info)
    ccall((:magma_zgeqp3, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), m, n, A, lda, jpvt, tau, work, lwork, rwork, info)
end

"""
    magma_zgeqp3_gpu(m, n, dA, ldda, jpvt, tau, dwork, lwork, rwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqp3_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *jpvt, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dwork, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_zgeqp3_gpu(m, n, dA, ldda, jpvt, tau, dwork, lwork, rwork, info)
    ccall((:magma_zgeqp3_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), m, n, dA, ldda, jpvt, tau, dwork, lwork, rwork, info)
end

"""
    magma_zgeqr2_gpu(m, n, dA, ldda, dtau, dwork, queue, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqr2_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dtau, magmaDouble_ptr dwork, magma_queue_t queue, magma_int_t *info);
```
"""
function magma_zgeqr2_gpu(m, n, dA, ldda, dtau, dwork, queue, info)
    ccall((:magma_zgeqr2_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDouble_ptr, magma_queue_t, Ptr{magma_int_t}), m, n, dA, ldda, dtau, dwork, queue, info)
end

"""
    magma_zgeqr2x_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqr2x_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dtau, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr ddA, magmaDouble_ptr dwork, magma_int_t *info);
```
"""
function magma_zgeqr2x_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, info)
    ccall((:magma_zgeqr2x_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, Ptr{magma_int_t}), m, n, dA, ldda, dtau, dT, ddA, dwork, info)
end

"""
    magma_zgeqr2x2_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqr2x2_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dtau, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr ddA, magmaDouble_ptr dwork, magma_int_t *info);
```
"""
function magma_zgeqr2x2_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, info)
    ccall((:magma_zgeqr2x2_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, Ptr{magma_int_t}), m, n, dA, ldda, dtau, dT, ddA, dwork, info)
end

"""
    magma_zgeqr2x3_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, info)


### Prototype
```c
magma_int_t magma_zgeqr2x3_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dtau, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr ddA, magmaDouble_ptr dwork, magma_int_t *info);
```
"""
function magma_zgeqr2x3_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, info)
    ccall((:magma_zgeqr2x3_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, Ptr{magma_int_t}), m, n, dA, ldda, dtau, dT, ddA, dwork, info)
end

"""
    magma_zgeqr2x4_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, queue, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqr2x4_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dtau, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr ddA, magmaDouble_ptr dwork, magma_queue_t queue, magma_int_t *info);
```
"""
function magma_zgeqr2x4_gpu(m, n, dA, ldda, dtau, dT, ddA, dwork, queue, info)
    ccall((:magma_zgeqr2x4_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magmaDouble_ptr, magma_queue_t, Ptr{magma_int_t}), m, n, dA, ldda, dtau, dT, ddA, dwork, queue, info)
end

"""
    magma_zgeqrf(m, n, A, lda, tau, work, lwork, info)


### Prototype
```c
magma_int_t magma_zgeqrf( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgeqrf(m, n, A, lda, tau, work, lwork, info)
    ccall((:magma_zgeqrf, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, tau, work, lwork, info)
end

"""
    magma_zgeqrf_gpu(m, n, dA, ldda, tau, dT, info)


### Prototype
```c
magma_int_t magma_zgeqrf_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t *info);
```
"""
function magma_zgeqrf_gpu(m, n, dA, ldda, tau, dT, info)
    ccall((:magma_zgeqrf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, Ptr{magma_int_t}), m, n, dA, ldda, tau, dT, info)
end

"""
    magma_zgeqrf_m(ngpu, m, n, A, lda, tau, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqrf_m( magma_int_t ngpu, magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgeqrf_m(ngpu, m, n, A, lda, tau, work, lwork, info)
    ccall((:magma_zgeqrf_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, m, n, A, lda, tau, work, lwork, info)
end

"""
    magma_zgeqrf_ooc(m, n, A, lda, tau, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqrf_ooc( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgeqrf_ooc(m, n, A, lda, tau, work, lwork, info)
    ccall((:magma_zgeqrf_ooc, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, tau, work, lwork, info)
end

"""
    magma_zgeqrf2_gpu(m, n, dA, ldda, tau, info)


### Prototype
```c
magma_int_t magma_zgeqrf2_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magma_int_t *info);
```
"""
function magma_zgeqrf2_gpu(m, n, dA, ldda, tau, info)
    ccall((:magma_zgeqrf2_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}), m, n, dA, ldda, tau, info)
end

"""
    magma_zgeqrf2_mgpu(ngpu, m, n, d_lA, ldda, tau, info)


### Prototype
```c
magma_int_t magma_zgeqrf2_mgpu( magma_int_t ngpu, magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr d_lA[], magma_int_t ldda, magmaDoubleComplex *tau, magma_int_t *info);
```
"""
function magma_zgeqrf2_mgpu(ngpu, m, n, d_lA, ldda, tau, info)
    ccall((:magma_zgeqrf2_mgpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}), ngpu, m, n, d_lA, ldda, tau, info)
end

"""
    magma_zgeqrf3_gpu(m, n, dA, ldda, tau, dT, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqrf3_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t *info);
```
"""
function magma_zgeqrf3_gpu(m, n, dA, ldda, tau, dT, info)
    ccall((:magma_zgeqrf3_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, Ptr{magma_int_t}), m, n, dA, ldda, tau, dT, info)
end

"""
    magma_zgeqrs_gpu(m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info)


### Prototype
```c
magma_int_t magma_zgeqrs_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex const *tau, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex *hwork, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgeqrs_gpu(m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info)
    ccall((:magma_zgeqrs_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info)
end

"""
    magma_zgeqrs3_gpu(m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgeqrs3_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex const *tau, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex *hwork, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgeqrs3_gpu(m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info)
    ccall((:magma_zgeqrs3_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, nrhs, dA, ldda, tau, dT, dB, lddb, hwork, lwork, info)
end

"""
    magma_zgerbt_gpu(gen, n, nrhs, dA, ldda, dB, lddb, U, V, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgerbt_gpu( magma_bool_t gen, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex *U, magmaDoubleComplex *V, magma_int_t *info);
```
"""
function magma_zgerbt_gpu(gen, n, nrhs, dA, ldda, dB, lddb, U, V, info)
    ccall((:magma_zgerbt_gpu, libmagma), magma_int_t, (magma_bool_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}), gen, n, nrhs, dA, ldda, dB, lddb, U, V, info)
end

"""
    magma_zgerfs_nopiv_gpu(trans, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dAF, iter, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgerfs_nopiv_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magmaDoubleComplex_ptr dworkd, magmaDoubleComplex_ptr dAF, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_zgerfs_nopiv_gpu(trans, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dAF, iter, info)
    ccall((:magma_zgerfs_nopiv_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, Ptr{magma_int_t}, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dAF, iter, info)
end

"""
    magma_zgesdd(jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, rwork, iwork, info)


### Prototype
```c
magma_int_t magma_zgesdd( magma_vec_t jobz, magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *s, magmaDoubleComplex *U, magma_int_t ldu, magmaDoubleComplex *VT, magma_int_t ldvt, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *iwork, magma_int_t *info);
```
"""
function magma_zgesdd(jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, rwork, iwork, info)
    ccall((:magma_zgesdd, libmagma), magma_int_t, (magma_vec_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}, Ptr{magma_int_t}), jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, rwork, iwork, info)
end

"""
    magma_zgesv(n, nrhs, A, lda, ipiv, B, ldb, info)


### Prototype
```c
magma_int_t magma_zgesv( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);
```
"""
function magma_zgesv(n, nrhs, A, lda, ipiv, B, ldb, info)
    ccall((:magma_zgesv, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), n, nrhs, A, lda, ipiv, B, ldb, info)
end

"""
    magma_zgesv_gpu(n, nrhs, dA, ldda, ipiv, dB, lddb, info)


### Prototype
```c
magma_int_t magma_zgesv_gpu( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zgesv_gpu(n, nrhs, dA, ldda, ipiv, dB, lddb, info)
    ccall((:magma_zgesv_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), n, nrhs, dA, ldda, ipiv, dB, lddb, info)
end

"""
    magma_zgesv_nopiv_gpu(n, nrhs, dA, ldda, dB, lddb, info)


### Prototype
```c
magma_int_t magma_zgesv_nopiv_gpu( magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zgesv_nopiv_gpu(n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zgesv_nopiv_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zgesv_rbt(ref, n, nrhs, A, lda, B, ldb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgesv_rbt( magma_bool_t ref, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);
```
"""
function magma_zgesv_rbt(ref, n, nrhs, A, lda, B, ldb, info)
    ccall((:magma_zgesv_rbt, libmagma), magma_int_t, (magma_bool_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ref, n, nrhs, A, lda, B, ldb, info)
end

"""
    magma_zgesvd(jobu, jobvt, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, rwork, info)


### Prototype
```c
magma_int_t magma_zgesvd( magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *s, magmaDoubleComplex *U, magma_int_t ldu, magmaDoubleComplex *VT, magma_int_t ldvt, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_zgesvd(jobu, jobvt, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, rwork, info)
    ccall((:magma_zgesvd, libmagma), magma_int_t, (magma_vec_t, magma_vec_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), jobu, jobvt, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, rwork, info)
end

"""
    magma_zgetf2_gpu(m, n, dA, ldda, ipiv, queue, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetf2_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_queue_t queue, magma_int_t *info);
```
"""
function magma_zgetf2_gpu(m, n, dA, ldda, ipiv, queue, info)
    ccall((:magma_zgetf2_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magma_queue_t, Ptr{magma_int_t}), m, n, dA, ldda, ipiv, queue, info)
end

"""
    magma_zgetf2_native_fused(m, n, dA, ldda, ipiv, gbstep, flags, info, queue)


### Prototype
```c
magma_int_t magma_zgetf2_native_fused( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t gbstep, magma_int_t *flags, magma_int_t *info, magma_queue_t queue );
```
"""
function magma_zgetf2_native_fused(m, n, dA, ldda, ipiv, gbstep, flags, info, queue)
    ccall((:magma_zgetf2_native_fused, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_queue_t), m, n, dA, ldda, ipiv, gbstep, flags, info, queue)
end

"""
    magma_zgetf2_native(m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue)


### Prototype
```c
magma_int_t magma_zgetf2_native( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *dipiv, magma_int_t* dipivinfo, magma_int_t *dinfo, magma_int_t gbstep, magma_queue_t queue, magma_queue_t update_queue);
```
"""
function magma_zgetf2_native(m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue)
    ccall((:magma_zgetf2_native, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t, magma_queue_t), m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue)
end

"""
    magma_zgetf2_nopiv(m, n, A, lda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetf2_nopiv( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zgetf2_nopiv(m, n, A, lda, info)
    ccall((:magma_zgetf2_nopiv, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, info)
end

"""
    magma_zgetrf_recpanel_native(m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue)


### Prototype
```c
magma_int_t magma_zgetrf_recpanel_native( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t* dipiv, magma_int_t* dipivinfo, magma_int_t *dinfo, magma_int_t gbstep, magma_queue_t queue, magma_queue_t update_queue );
```
"""
function magma_zgetrf_recpanel_native(m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue)
    ccall((:magma_zgetrf_recpanel_native, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_queue_t, magma_queue_t), m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue)
end

"""
    magma_zgetrf(m, n, A, lda, ipiv, info)


### Prototype
```c
magma_int_t magma_zgetrf( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zgetrf(m, n, A, lda, ipiv, info)
    ccall((:magma_zgetrf, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, A, lda, ipiv, info)
end

"""
    magma_zgetrf_gpu(m, n, dA, ldda, ipiv, info)


### Prototype
```c
magma_int_t magma_zgetrf_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zgetrf_gpu(m, n, dA, ldda, ipiv, info)
    ccall((:magma_zgetrf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, dA, ldda, ipiv, info)
end

"""
    magma_zgetrf_gpu_expert(m, n, dA, ldda, ipiv, info, nb, mode)


### Prototype
```c
magma_int_t magma_zgetrf_gpu_expert( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info, magma_int_t nb, magma_mode_t mode);
```
"""
function magma_zgetrf_gpu_expert(m, n, dA, ldda, ipiv, info, nb, mode)
    ccall((:magma_zgetrf_gpu_expert, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_int_t, magma_mode_t), m, n, dA, ldda, ipiv, info, nb, mode)
end

"""
    magma_zgetrf_native(m, n, dA, ldda, ipiv, info)


### Prototype
```c
magma_int_t magma_zgetrf_native( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info );
```
"""
function magma_zgetrf_native(m, n, dA, ldda, ipiv, info)
    ccall((:magma_zgetrf_native, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, dA, ldda, ipiv, info)
end

"""
    magma_zgetrf_m(ngpu, m, n, A, lda, ipiv, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetrf_m( magma_int_t ngpu, magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zgetrf_m(ngpu, m, n, A, lda, ipiv, info)
    ccall((:magma_zgetrf_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), ngpu, m, n, A, lda, ipiv, info)
end

"""
    magma_zgetrf_mgpu(ngpu, m, n, d_lA, ldda, ipiv, info)


### Prototype
```c
magma_int_t magma_zgetrf_mgpu( magma_int_t ngpu, magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr d_lA[], magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zgetrf_mgpu(ngpu, m, n, d_lA, ldda, ipiv, info)
    ccall((:magma_zgetrf_mgpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), ngpu, m, n, d_lA, ldda, ipiv, info)
end

"""
    magma_zgetrf2(m, n, A, lda, ipiv, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetrf2( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zgetrf2(m, n, A, lda, ipiv, info)
    ccall((:magma_zgetrf2, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, A, lda, ipiv, info)
end

"""
    magma_zgetrf2_mgpu(ngpu, m, n, nb, offset, d_lAT, lddat, ipiv, d_lAP, W, ldw, queues, info)


### Prototype
```c
magma_int_t magma_zgetrf2_mgpu( magma_int_t ngpu, magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset, magmaDoubleComplex_ptr d_lAT[], magma_int_t lddat, magma_int_t *ipiv, magmaDoubleComplex_ptr d_lAP[], magmaDoubleComplex *W, magma_int_t ldw, magma_queue_t queues[][2], magma_int_t *info);
```
"""
function magma_zgetrf2_mgpu(ngpu, m, n, nb, offset, d_lAT, lddat, ipiv, d_lAP, W, ldw, queues, info)
    ccall((:magma_zgetrf2_mgpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex_ptr}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{NTuple{2, magma_queue_t}}, Ptr{magma_int_t}), ngpu, m, n, nb, offset, d_lAT, lddat, ipiv, d_lAP, W, ldw, queues, info)
end

"""
    magma_zgetrf_nopiv(m, n, A, lda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetrf_nopiv( magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zgetrf_nopiv(m, n, A, lda, info)
    ccall((:magma_zgetrf_nopiv, libmagma), magma_int_t, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, A, lda, info)
end

"""
    magma_zgetrf_nopiv_gpu(m, n, dA, ldda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetrf_nopiv_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zgetrf_nopiv_gpu(m, n, dA, ldda, info)
    ccall((:magma_zgetrf_nopiv_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, dA, ldda, info)
end

"""
    magma_zgetri_gpu(n, dA, ldda, ipiv, dwork, lwork, info)


### Prototype
```c
magma_int_t magma_zgetri_gpu( magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaDoubleComplex_ptr dwork, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zgetri_gpu(n, dA, ldda, ipiv, dwork, lwork, info)
    ccall((:magma_zgetri_gpu, libmagma), magma_int_t, (magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), n, dA, ldda, ipiv, dwork, lwork, info)
end

"""
    magma_zgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)


### Prototype
```c
magma_int_t magma_zgetrs_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
    ccall((:magma_zgetrs_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
end

"""
    magma_zgetrs_nopiv_gpu(trans, n, nrhs, dA, ldda, dB, lddb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zgetrs_nopiv_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zgetrs_nopiv_gpu(trans, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zgetrs_nopiv_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zheevd(jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)

------------------------------------------------------------ zhe routines
### Prototype
```c
magma_int_t magma_zheevd( magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevd(jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevd, libmagma), magma_int_t, (magma_vec_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevd_gpu( magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, double *w, magmaDoubleComplex *wA, magma_int_t ldwa, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevd_gpu, libmagma), magma_int_t, (magma_vec_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevd_m(ngpu, jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevd_m( magma_int_t ngpu, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevd_m(ngpu, jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevd_m, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevdx(jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevdx( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevdx(jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevdx, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevdx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevdx_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *wA, magma_int_t ldwa, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevdx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevdx_gpu, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, mout, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevdx_m(ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevdx_m( magma_int_t ngpu, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevdx_m(ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevdx_m, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevdx_2stage(jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevdx_2stage( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevdx_2stage(jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevdx_2stage, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevdx_2stage_m(ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevdx_2stage_m( magma_int_t ngpu, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevdx_2stage_m(ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevdx_2stage_m, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevr(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, mout, w, Z, ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)

no real [sd] precisions available
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevr( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout, double *w, magmaDoubleComplex *Z, magma_int_t ldz, magma_int_t *isuppz, magmaDoubleComplex *work, magma_int_t lwork, double *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevr(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, mout, w, Z, ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevr, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Cdouble, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, mout, w, Z, ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevr_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol, mout, w, dZ, lddz, isuppz, wA, ldwa, wZ, ldwz, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevr_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, double vl, double vu, magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout, double *w, magmaDoubleComplex_ptr dZ, magma_int_t lddz, magma_int_t *isuppz, magmaDoubleComplex *wA, magma_int_t ldwa, magmaDoubleComplex *wZ, magma_int_t ldwz, magmaDoubleComplex *work, magma_int_t lwork, double *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zheevr_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol, mout, w, dZ, lddz, isuppz, wA, ldwa, wZ, ldwz, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zheevr_gpu, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Cdouble, Ptr{magma_int_t}, Ptr{Cdouble}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol, mout, w, dZ, lddz, isuppz, wA, ldwa, wZ, ldwz, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zheevx(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, mout, w, Z, ldz, work, lwork, rwork, iwork, ifail, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevx( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double vl, double vu, magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout, double *w, magmaDoubleComplex *Z, magma_int_t ldz, magmaDoubleComplex *work, magma_int_t lwork, double *rwork, magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);
```
"""
function magma_zheevx(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, mout, w, Z, ldz, work, lwork, rwork, iwork, ifail, info)
    ccall((:magma_zheevx, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Cdouble, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, mout, w, Z, ldz, work, lwork, rwork, iwork, ifail, info)
end

"""
    magma_zheevx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol, mout, w, dZ, lddz, wA, ldwa, wZ, ldwz, work, lwork, rwork, iwork, ifail, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zheevx_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, double vl, double vu, magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout, double *w, magmaDoubleComplex_ptr dZ, magma_int_t lddz, magmaDoubleComplex *wA, magma_int_t ldwa, magmaDoubleComplex *wZ, magma_int_t ldwz, magmaDoubleComplex *work, magma_int_t lwork, double *rwork, magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);
```
"""
function magma_zheevx_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol, mout, w, dZ, lddz, wA, ldwa, wZ, ldwz, work, lwork, rwork, iwork, ifail, info)
    ccall((:magma_zheevx_gpu, libmagma), magma_int_t, (magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Cdouble, Ptr{magma_int_t}, Ptr{Cdouble}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol, mout, w, dZ, lddz, wA, ldwa, wZ, ldwz, work, lwork, rwork, iwork, ifail, info)
end

"""
    magma_zhegst(itype, uplo, n, A, lda, B, ldb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegst( magma_int_t itype, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);
```
"""
function magma_zhegst(itype, uplo, n, A, lda, B, ldb, info)
    ccall((:magma_zhegst, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), itype, uplo, n, A, lda, B, ldb, info)
end

"""
    magma_zhegst_gpu(itype, uplo, n, dA, ldda, dB, lddb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegst_gpu( magma_int_t itype, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_const_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zhegst_gpu(itype, uplo, n, dA, ldda, dB, lddb, info)
    ccall((:magma_zhegst_gpu, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magma_int_t}), itype, uplo, n, dA, ldda, dB, lddb, info)
end

"""
    magma_zhegst_m(ngpu, itype, uplo, n, A, lda, B, ldb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegst_m( magma_int_t ngpu, magma_int_t itype, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);
```
"""
function magma_zhegst_m(ngpu, itype, uplo, n, A, lda, B, ldb, info)
    ccall((:magma_zhegst_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, itype, uplo, n, A, lda, B, ldb, info)
end

"""
    magma_zhegvd(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvd( magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvd(itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvd, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvd_m(ngpu, itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvd_m( magma_int_t ngpu, magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvd_m(ngpu, itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvd_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_vec_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, itype, jobz, uplo, n, A, lda, B, ldb, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvdx(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvdx( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvdx(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvdx, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvdx_m(ngpu, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvdx_m( magma_int_t ngpu, magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvdx_m(ngpu, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvdx_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvdx_2stage(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvdx_2stage( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvdx_2stage(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvdx_2stage, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvdx_2stage_m(ngpu, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvdx_2stage_m( magma_int_t ngpu, magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double vl, double vu, magma_int_t il, magma_int_t iu, magma_int_t *mout, double *w, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, magma_int_t lrwork, #endif magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvdx_2stage_m(ngpu, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvdx_2stage_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, mout, w, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvr(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, mout, w, Z, ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)

no real [sd] precisions available
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvr( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double vl, double vu, magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout, double *w, magmaDoubleComplex *Z, magma_int_t ldz, magma_int_t *isuppz, magmaDoubleComplex *work, magma_int_t lwork, double *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zhegvr(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, mout, w, Z, ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zhegvr, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Cdouble, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, mout, w, Z, ldz, isuppz, work, lwork, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_zhegvx(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, mout, w, Z, ldz, work, lwork, rwork, iwork, ifail, info)

no real [sd] precisions available
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhegvx( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, double vl, double vu, magma_int_t il, magma_int_t iu, double abstol, magma_int_t *mout, double *w, magmaDoubleComplex *Z, magma_int_t ldz, magmaDoubleComplex *work, magma_int_t lwork, double *rwork, magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);
```
"""
function magma_zhegvx(itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, mout, w, Z, ldz, work, lwork, rwork, iwork, ifail, info)
    ccall((:magma_zhegvx, libmagma), magma_int_t, (magma_int_t, magma_vec_t, magma_range_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Cdouble, Ptr{magma_int_t}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, mout, w, Z, ldz, work, lwork, rwork, iwork, ifail, info)
end

"""
    magma_zhesv(uplo, n, nrhs, A, lda, ipiv, B, ldb, info)


### Prototype
```c
magma_int_t magma_zhesv( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);
```
"""
function magma_zhesv(uplo, n, nrhs, A, lda, ipiv, B, ldb, info)
    ccall((:magma_zhesv, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, A, lda, ipiv, B, ldb, info)
end

"""
    magma_zhesv_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhesv_nopiv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zhesv_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zhesv_nopiv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zhetrd(uplo, n, A, lda, d, e, tau, work, lwork, info)


### Prototype
```c
magma_int_t magma_zhetrd( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *d, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zhetrd(uplo, n, A, lda, d, e, tau, work, lwork, info)
    ccall((:magma_zhetrd, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, A, lda, d, e, tau, work, lwork, info)
end

"""
    magma_zhetrd_gpu(uplo, n, dA, ldda, d, e, tau, wA, ldwa, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrd_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, double *d, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *wA, magma_int_t ldwa, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zhetrd_gpu(uplo, n, dA, ldda, d, e, tau, wA, ldwa, work, lwork, info)
    ccall((:magma_zhetrd_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, d, e, tau, wA, ldwa, work, lwork, info)
end

"""
    magma_zhetrd2_gpu(uplo, n, dA, ldda, d, e, tau, wA, ldwa, work, lwork, dwork, ldwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrd2_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, double *d, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *wA, magma_int_t ldwa, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex_ptr dwork, magma_int_t ldwork, magma_int_t *info);
```
"""
function magma_zhetrd2_gpu(uplo, n, dA, ldda, d, e, tau, wA, ldwa, work, lwork, dwork, ldwork, info)
    ccall((:magma_zhetrd2_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, d, e, tau, wA, ldwa, work, lwork, dwork, ldwork, info)
end

"""
    magma_zhetrd_mgpu(ngpu, nqueue, uplo, n, A, lda, d, e, tau, work, lwork, info)

TODO: rename magma_zhetrd_m?
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrd_mgpu( magma_int_t ngpu, magma_int_t nqueue, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, double *d, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zhetrd_mgpu(ngpu, nqueue, uplo, n, A, lda, d, e, tau, work, lwork, info)
    ccall((:magma_zhetrd_mgpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, nqueue, uplo, n, A, lda, d, e, tau, work, lwork, info)
end

"""
    magma_zhetrd_hb2st(uplo, n, nb, Vblksiz, A, lda, d, e, V, ldv, TAU, compT, T, ldt)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrd_hb2st( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz, magmaDoubleComplex *A, magma_int_t lda, double *d, double *e, magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU, magma_int_t compT, magmaDoubleComplex *T, magma_int_t ldt);
```
"""
function magma_zhetrd_hb2st(uplo, n, nb, Vblksiz, A, lda, d, e, V, ldv, TAU, compT, T, ldt)
    ccall((:magma_zhetrd_hb2st, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t), uplo, n, nb, Vblksiz, A, lda, d, e, V, ldv, TAU, compT, T, ldt)
end

"""
    magma_zhetrd_he2hb(uplo, n, nb, A, lda, tau, work, lwork, dT, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrd_he2hb( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex_ptr dT, magma_int_t *info);
```
"""
function magma_zhetrd_he2hb(uplo, n, nb, A, lda, tau, work, lwork, dT, info)
    ccall((:magma_zhetrd_he2hb, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, Ptr{magma_int_t}), uplo, n, nb, A, lda, tau, work, lwork, dT, info)
end

"""
    magma_zhetrd_he2hb_mgpu(uplo, n, nb, A, lda, tau, work, lwork, dAmgpu, ldda, dTmgpu, lddt, ngpu, distblk, queues, nqueue, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrd_he2hb_mgpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex_ptr dAmgpu[], magma_int_t ldda, magmaDoubleComplex_ptr dTmgpu[], magma_int_t lddt, magma_int_t ngpu, magma_int_t distblk, magma_queue_t queues[][20], magma_int_t nqueue, magma_int_t *info);
```
"""
function magma_zhetrd_he2hb_mgpu(uplo, n, nb, A, lda, tau, work, lwork, dAmgpu, ldda, dTmgpu, lddt, ngpu, distblk, queues, nqueue, info)
    ccall((:magma_zhetrd_he2hb_mgpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, magma_int_t, Ptr{NTuple{20, magma_queue_t}}, magma_int_t, Ptr{magma_int_t}), uplo, n, nb, A, lda, tau, work, lwork, dAmgpu, ldda, dTmgpu, lddt, ngpu, distblk, queues, nqueue, info)
end

"""
    magma_zhetrf(uplo, n, A, lda, ipiv, info)


### Prototype
```c
magma_int_t magma_zhetrf( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zhetrf(uplo, n, A, lda, ipiv, info)
    ccall((:magma_zhetrf, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, n, A, lda, ipiv, info)
end

"""
    magma_zhetrf_gpu(uplo, n, dA, ldda, ipiv, info)


### Prototype
```c
magma_int_t magma_zhetrf_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zhetrf_gpu(uplo, n, dA, ldda, ipiv, info)
    ccall((:magma_zhetrf_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, n, dA, ldda, ipiv, info)
end

"""
    magma_zhetrf_aasen(uplo, cpu_panel, n, A, lda, ipiv, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrf_aasen( magma_uplo_t uplo, magma_int_t cpu_panel, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);
```
"""
function magma_zhetrf_aasen(uplo, cpu_panel, n, A, lda, ipiv, info)
    ccall((:magma_zhetrf_aasen, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, cpu_panel, n, A, lda, ipiv, info)
end

"""
    magma_zhetrf_nopiv(uplo, n, A, lda, info)


### Prototype
```c
magma_int_t magma_zhetrf_nopiv( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zhetrf_nopiv(uplo, n, A, lda, info)
    ccall((:magma_zhetrf_nopiv, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, A, lda, info)
end

"""
    magma_zhetrf_nopiv_cpu(uplo, n, ib, A, lda, info)


### Prototype
```c
magma_int_t magma_zhetrf_nopiv_cpu( magma_uplo_t uplo, magma_int_t n, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zhetrf_nopiv_cpu(uplo, n, ib, A, lda, info)
    ccall((:magma_zhetrf_nopiv_cpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, ib, A, lda, info)
end

"""
    magma_zhetrf_nopiv_gpu(uplo, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_zhetrf_nopiv_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zhetrf_nopiv_gpu(uplo, n, dA, ldda, info)
    ccall((:magma_zhetrf_nopiv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_zhetrs_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zhetrs_nopiv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zhetrs_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zhetrs_nopiv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zlabrd_gpu(m, n, nb, A, lda, dA, ldda, d, e, tauq, taup, X, ldx, dX, lddx, Y, ldy, dY, lddy, work, lwork, queue)


### Prototype
```c
magma_int_t magma_zlabrd_gpu( magma_int_t m, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex_ptr dA, magma_int_t ldda, double *d, double *e, magmaDoubleComplex *tauq, magmaDoubleComplex *taup, magmaDoubleComplex *X, magma_int_t ldx, magmaDoubleComplex_ptr dX, magma_int_t lddx, magmaDoubleComplex *Y, magma_int_t ldy, magmaDoubleComplex_ptr dY, magma_int_t lddy, magmaDoubleComplex *work, magma_int_t lwork, magma_queue_t queue);
```
"""
function magma_zlabrd_gpu(m, n, nb, A, lda, dA, ldda, d, e, tauq, taup, X, ldx, dX, lddx, Y, ldy, dY, lddy, work, lwork, queue)
    ccall((:magma_zlabrd_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), m, n, nb, A, lda, dA, ldda, d, e, tauq, taup, X, ldx, dX, lddx, Y, ldy, dY, lddy, work, lwork, queue)
end

"""
    magma_zlahef_gpu(uplo, n, nb, kb, dA, ldda, ipiv, dW, lddw, queues, info)


### Prototype
```c
magma_int_t magma_zlahef_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaDoubleComplex_ptr dW, magma_int_t lddw, magma_queue_t queues[], magma_int_t *info);
```
"""
function magma_zlahef_gpu(uplo, n, nb, kb, dA, ldda, ipiv, dW, lddw, queues, info)
    ccall((:magma_zlahef_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_queue_t}, Ptr{magma_int_t}), uplo, n, nb, kb, dA, ldda, ipiv, dW, lddw, queues, info)
end

"""
    magma_zlahr2(n, k, nb, dA, ldda, dV, lddv, A, lda, tau, T, ldt, Y, ldy, queue)


### Prototype
```c
magma_int_t magma_zlahr2( magma_int_t n, magma_int_t k, magma_int_t nb, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dV, magma_int_t lddv, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *T, magma_int_t ldt, magmaDoubleComplex *Y, magma_int_t ldy, magma_queue_t queue);
```
"""
function magma_zlahr2(n, k, nb, dA, ldda, dV, lddv, A, lda, tau, T, ldt, Y, ldy, queue)
    ccall((:magma_zlahr2, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), n, k, nb, dA, ldda, dV, lddv, A, lda, tau, T, ldt, Y, ldy, queue)
end

"""
    magma_zlahr2_m(n, k, nb, A, lda, tau, T, ldt, Y, ldy, data)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlahr2_m( magma_int_t n, magma_int_t k, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *T, magma_int_t ldt, magmaDoubleComplex *Y, magma_int_t ldy, struct zgehrd_data *data);
```
"""
function magma_zlahr2_m(n, k, nb, A, lda, tau, T, ldt, Y, ldy, data)
    ccall((:magma_zlahr2_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{zgehrd_data}), n, k, nb, A, lda, tau, T, ldt, Y, ldy, data)
end

"""
    magma_zlahru(n, ihi, k, nb, A, lda, dA, ldda, dY, lddy, dV, lddv, dT, dwork, queue)


### Prototype
```c
magma_int_t magma_zlahru( magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dY, magma_int_t lddy, magmaDoubleComplex_ptr dV, magma_int_t lddv, magmaDoubleComplex_ptr dT, magmaDoubleComplex_ptr dwork, magma_queue_t queue);
```
"""
function magma_zlahru(n, ihi, k, nb, A, lda, dA, ldda, dY, lddy, dV, lddv, dT, dwork, queue)
    ccall((:magma_zlahru, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_queue_t), n, ihi, k, nb, A, lda, dA, ldda, dY, lddy, dV, lddv, dT, dwork, queue)
end

"""
    magma_zlahru_m(n, ihi, k, nb, A, lda, data)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlahru_m( magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, struct zgehrd_data *data);
```
"""
function magma_zlahru_m(n, ihi, k, nb, A, lda, data)
    ccall((:magma_zlahru_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{zgehrd_data}), n, ihi, k, nb, A, lda, data)
end

"""
    magma_zlaqps(m, n, offset, nb, kb, A, lda, dA, ldda, jpvt, tau, vn1, vn2, auxv, F, ldf, dF, lddf)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlaqps( magma_int_t m, magma_int_t n, magma_int_t offset, magma_int_t nb, magma_int_t *kb, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *jpvt, magmaDoubleComplex *tau, double *vn1, double *vn2, magmaDoubleComplex *auxv, magmaDoubleComplex *F, magma_int_t ldf, magmaDoubleComplex_ptr dF, magma_int_t lddf);
```
"""
function magma_zlaqps(m, n, offset, nb, kb, A, lda, dA, ldda, jpvt, tau, vn1, vn2, auxv, F, ldf, dF, lddf)
    ccall((:magma_zlaqps, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t), m, n, offset, nb, kb, A, lda, dA, ldda, jpvt, tau, vn1, vn2, auxv, F, ldf, dF, lddf)
end

"""
    magma_zlaqps_gpu(m, n, offset, nb, kb, dA, ldda, jpvt, tau, vn1, vn2, dauxv, dF, lddf)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlaqps_gpu( magma_int_t m, magma_int_t n, magma_int_t offset, magma_int_t nb, magma_int_t *kb, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *jpvt, magmaDoubleComplex *tau, double *vn1, double *vn2, magmaDoubleComplex_ptr dauxv, magmaDoubleComplex_ptr dF, magma_int_t lddf);
```
"""
function magma_zlaqps_gpu(m, n, offset, nb, kb, dA, ldda, jpvt, tau, vn1, vn2, dauxv, dF, lddf)
    ccall((:magma_zlaqps_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, Ptr{Cdouble}, Ptr{Cdouble}, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t), m, n, offset, nb, kb, dA, ldda, jpvt, tau, vn1, vn2, dauxv, dF, lddf)
end

"""
    magma_zlaqps2_gpu(m, n, offset, nb, kb, dA, ldda, jpvt, dtau, dvn1, dvn2, dauxv, dF, lddf, dlsticcs, queue)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlaqps2_gpu( magma_int_t m, magma_int_t n, magma_int_t offset, magma_int_t nb, magma_int_t *kb, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *jpvt, magmaDoubleComplex_ptr dtau, magmaDouble_ptr dvn1, magmaDouble_ptr dvn2, magmaDoubleComplex_ptr dauxv, magmaDoubleComplex_ptr dF, magma_int_t lddf, magmaDouble_ptr dlsticcs, magma_queue_t queue);
```
"""
function magma_zlaqps2_gpu(m, n, offset, nb, kb, dA, ldda, jpvt, dtau, dvn1, dvn2, dauxv, dF, lddf, dlsticcs, queue)
    ccall((:magma_zlaqps2_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magmaDoubleComplex_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDoubleComplex_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDouble_ptr, magma_queue_t), m, n, offset, nb, kb, dA, ldda, jpvt, dtau, dvn1, dvn2, dauxv, dF, lddf, dlsticcs, queue)
end

"""
    magma_zlarf_gpu(m, n, dv, dtau, dC, lddc, queue)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlarf_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dv, magmaDoubleComplex_const_ptr dtau, magmaDoubleComplex_ptr dC, magma_int_t lddc, magma_queue_t queue);
```
"""
function magma_zlarf_gpu(m, n, dv, dtau, dC, lddc, queue)
    ccall((:magma_zlarf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magmaDoubleComplex_const_ptr, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), m, n, dv, dtau, dC, lddc, queue)
end

"""
    magma_zlarfb2_gpu(m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork, queue)

in zgeqr2x_gpu-v3.cpp
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlarfb2_gpu( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_const_ptr dV, magma_int_t lddv, magmaDoubleComplex_const_ptr dT, magma_int_t lddt, magmaDoubleComplex_ptr dC, magma_int_t lddc, magmaDoubleComplex_ptr dwork, magma_int_t ldwork, magma_queue_t queue);
```
"""
function magma_zlarfb2_gpu(m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork, queue)
    ccall((:magma_zlarfb2_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), m, n, k, dV, lddv, dT, lddt, dC, lddc, dwork, ldwork, queue)
end

"""
    magma_zlatrd(uplo, n, nb, A, lda, e, tau, W, ldw, work, lwork, dA, ldda, dW, lddw, queue)


### Prototype
```c
magma_int_t magma_zlatrd( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *W, magma_int_t ldw, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dW, magma_int_t lddw, magma_queue_t queue);
```
"""
function magma_zlatrd(uplo, n, nb, A, lda, e, tau, W, ldw, work, lwork, dA, ldda, dW, lddw, queue)
    ccall((:magma_zlatrd, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, nb, A, lda, e, tau, W, ldw, work, lwork, dA, ldda, dW, lddw, queue)
end

"""
    magma_zlatrd2(uplo, n, nb, A, lda, e, tau, W, ldw, work, lwork, dA, ldda, dW, lddw, dwork, ldwork, queue)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlatrd2( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex *A, magma_int_t lda, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *W, magma_int_t ldw, magmaDoubleComplex *work, magma_int_t lwork, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dW, magma_int_t lddw, magmaDoubleComplex_ptr dwork, magma_int_t ldwork, magma_queue_t queue);
```
"""
function magma_zlatrd2(uplo, n, nb, A, lda, e, tau, W, ldw, work, lwork, dA, ldda, dW, lddw, dwork, ldwork, queue)
    ccall((:magma_zlatrd2, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t), uplo, n, nb, A, lda, e, tau, W, ldw, work, lwork, dA, ldda, dW, lddw, dwork, ldwork, queue)
end

"""
    magma_zlatrd_mgpu(ngpu, uplo, n, nb, nb0, A, lda, e, tau, W, ldw, dA, ldda, offset, dW, lddw, hwork, lhwork, dwork, ldwork, queues)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlatrd_mgpu( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t nb0, magmaDoubleComplex *A, magma_int_t lda, double *e, magmaDoubleComplex *tau, magmaDoubleComplex *W, magma_int_t ldw, magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t offset, magmaDoubleComplex_ptr dW[], magma_int_t lddw, magmaDoubleComplex *hwork, magma_int_t lhwork, magmaDoubleComplex_ptr dwork[], magma_int_t ldwork, magma_queue_t queues[]);
```
"""
function magma_zlatrd_mgpu(ngpu, uplo, n, nb, nb0, A, lda, e, tau, W, ldw, dA, ldda, offset, dW, lddw, hwork, lhwork, dwork, ldwork, queues)
    ccall((:magma_zlatrd_mgpu, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magma_queue_t}), ngpu, uplo, n, nb, nb0, A, lda, e, tau, W, ldw, dA, ldda, offset, dW, lddw, hwork, lhwork, dwork, ldwork, queues)
end

"""
    magma_zlatrsd(uplo, trans, diag, normin, n, A, lda, lambda, x, scale, cnorm, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zlatrsd( magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_bool_t normin, magma_int_t n, const magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex lambda, magmaDoubleComplex *x, double *scale, double *cnorm, magma_int_t *info);
```
"""
function magma_zlatrsd(uplo, trans, diag, normin, n, A, lda, lambda, x, scale, cnorm, info)
    ccall((:magma_zlatrsd, libmagma), magma_int_t, (magma_uplo_t, magma_trans_t, magma_diag_t, magma_bool_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magma_int_t}), uplo, trans, diag, normin, n, A, lda, lambda, x, scale, cnorm, info)
end

"""
    magma_zlauum(uplo, n, A, lda, info)


### Prototype
```c
magma_int_t magma_zlauum( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zlauum(uplo, n, A, lda, info)
    ccall((:magma_zlauum, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, A, lda, info)
end

"""
    magma_zlauum_gpu(uplo, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_zlauum_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zlauum_gpu(uplo, n, dA, ldda, info)
    ccall((:magma_zlauum_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_zposv(uplo, n, nrhs, A, lda, B, ldb, info)

------------------------------------------------------------ zpo routines
### Prototype
```c
magma_int_t magma_zposv( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);
```
"""
function magma_zposv(uplo, n, nrhs, A, lda, B, ldb, info)
    ccall((:magma_zposv, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, A, lda, B, ldb, info)
end

"""
    magma_zposv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)


### Prototype
```c
magma_int_t magma_zposv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zposv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zposv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zpotf2_gpu(uplo, n, dA, ldda, queue, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zpotf2_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_queue_t queue, magma_int_t *info);
```
"""
function magma_zpotf2_gpu(uplo, n, dA, ldda, queue, info)
    ccall((:magma_zpotf2_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magma_queue_t, Ptr{magma_int_t}), uplo, n, dA, ldda, queue, info)
end

"""
    magma_zpotrf_rectile_native(uplo, n, recnb, dA, ldda, gbstep, dinfo, info, queue)


### Prototype
```c
magma_int_t magma_zpotrf_rectile_native( magma_uplo_t uplo, magma_int_t n, magma_int_t recnb, magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t gbstep, magma_int_t *dinfo, magma_int_t *info, magma_queue_t queue);
```
"""
function magma_zpotrf_rectile_native(uplo, n, recnb, dA, ldda, gbstep, dinfo, info, queue)
    ccall((:magma_zpotrf_rectile_native, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_queue_t), uplo, n, recnb, dA, ldda, gbstep, dinfo, info, queue)
end

"""
    magma_zpotrf(uplo, n, A, lda, info)


### Prototype
```c
magma_int_t magma_zpotrf( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zpotrf(uplo, n, A, lda, info)
    ccall((:magma_zpotrf, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, A, lda, info)
end

"""
    magma_zpotrf_expert_gpu(uplo, n, dA, ldda, info, nb, mode)


### Prototype
```c
magma_int_t magma_zpotrf_expert_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info, magma_int_t nb, magma_mode_t mode );
```
"""
function magma_zpotrf_expert_gpu(uplo, n, dA, ldda, info, nb, mode)
    ccall((:magma_zpotrf_expert_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_mode_t), uplo, n, dA, ldda, info, nb, mode)
end

"""
    magma_zpotrf_gpu(uplo, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_zpotrf_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zpotrf_gpu(uplo, n, dA, ldda, info)
    ccall((:magma_zpotrf_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_zpotrf_native(uplo, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_zpotrf_native( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magma_zpotrf_native(uplo, n, dA, ldda, info)
    ccall((:magma_zpotrf_native, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_zpotrf_m(ngpu, uplo, n, A, lda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zpotrf_m( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zpotrf_m(ngpu, uplo, n, A, lda, info)
    ccall((:magma_zpotrf_m, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, uplo, n, A, lda, info)
end

"""
    magma_zpotrf_mgpu(ngpu, uplo, n, d_lA, ldda, info)


### Prototype
```c
magma_int_t magma_zpotrf_mgpu( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr d_lA[], magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zpotrf_mgpu(ngpu, uplo, n, d_lA, ldda, info)
    ccall((:magma_zpotrf_mgpu, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magma_int_t}), ngpu, uplo, n, d_lA, ldda, info)
end

"""
    magma_zpotrf_mgpu_right(ngpu, uplo, n, d_lA, ldda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zpotrf_mgpu_right( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr d_lA[], magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zpotrf_mgpu_right(ngpu, uplo, n, d_lA, ldda, info)
    ccall((:magma_zpotrf_mgpu_right, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magma_int_t}), ngpu, uplo, n, d_lA, ldda, info)
end

"""
    magma_zpotrf3_mgpu(ngpu, uplo, m, n, off_i, off_j, nb, d_lA, ldda, d_lP, lddp, A, lda, h, queues, events, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zpotrf3_mgpu( magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magmaDoubleComplex_ptr d_lA[], magma_int_t ldda, magmaDoubleComplex_ptr d_lP[], magma_int_t lddp, magmaDoubleComplex *A, magma_int_t lda, magma_int_t h, magma_queue_t queues[][3], magma_event_t events[][5], magma_int_t *info);
```
"""
function magma_zpotrf3_mgpu(ngpu, uplo, m, n, off_i, off_j, nb, d_lA, ldda, d_lP, lddp, A, lda, h, queues, events, info)
    ccall((:magma_zpotrf3_mgpu, libmagma), magma_int_t, (magma_int_t, magma_uplo_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex_ptr}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{NTuple{3, magma_queue_t}}, Ptr{NTuple{5, magma_event_t}}, Ptr{magma_int_t}), ngpu, uplo, m, n, off_i, off_j, nb, d_lA, ldda, d_lP, lddp, A, lda, h, queues, events, info)
end

"""
    magma_zpotri(uplo, n, A, lda, info)


### Prototype
```c
magma_int_t magma_zpotri( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zpotri(uplo, n, A, lda, info)
    ccall((:magma_zpotri, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, A, lda, info)
end

"""
    magma_zpotri_gpu(uplo, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_zpotri_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zpotri_gpu(uplo, n, dA, ldda, info)
    ccall((:magma_zpotri_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_zpotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)


### Prototype
```c
magma_int_t magma_zpotrs_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zpotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zpotrs_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zsysv_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zsysv_nopiv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zsysv_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zsysv_nopiv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zsytrf_nopiv_cpu(uplo, n, ib, A, lda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zsytrf_nopiv_cpu( magma_uplo_t uplo, magma_int_t n, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_zsytrf_nopiv_cpu(uplo, n, ib, A, lda, info)
    ccall((:magma_zsytrf_nopiv_cpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, n, ib, A, lda, info)
end

"""
    magma_zsytrf_nopiv_gpu(uplo, n, dA, ldda, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zsytrf_nopiv_gpu( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_zsytrf_nopiv_gpu(uplo, n, dA, ldda, info)
    ccall((:magma_zsytrf_nopiv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_zsytrs_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zsytrs_nopiv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t *info);
```
"""
function magma_zsytrs_nopiv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info)
    ccall((:magma_zsytrs_nopiv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, info)
end

"""
    magma_zstedx(range, n, vl, vu, il, iu, d, e, Z, ldz, rwork, lrwork, iwork, liwork, dwork, info)

------------------------------------------------------------ zst routines
### Prototype
```c
magma_int_t magma_zstedx( magma_range_t range, magma_int_t n, double vl, double vu, magma_int_t il, magma_int_t iu, double *d, double *e, magmaDoubleComplex *Z, magma_int_t ldz, double *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magmaDouble_ptr dwork, magma_int_t *info);
```
"""
function magma_zstedx(range, n, vl, vu, il, iu, d, e, Z, ldz, rwork, lrwork, iwork, liwork, dwork, info)
    ccall((:magma_zstedx, libmagma), magma_int_t, (magma_range_t, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, magmaDouble_ptr, Ptr{magma_int_t}), range, n, vl, vu, il, iu, d, e, Z, ldz, rwork, lrwork, iwork, liwork, dwork, info)
end

"""
    magma_zstedx_m(ngpu, range, n, vl, vu, il, iu, d, e, Z, ldz, rwork, lrwork, iwork, liwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zstedx_m( magma_int_t ngpu, magma_range_t range, magma_int_t n, double vl, double vu, magma_int_t il, magma_int_t iu, double *d, double *e, magmaDoubleComplex *Z, magma_int_t ldz, double *rwork, magma_int_t lrwork, magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
```
"""
function magma_zstedx_m(ngpu, range, n, vl, vu, il, iu, d, e, Z, ldz, rwork, lrwork, iwork, liwork, info)
    ccall((:magma_zstedx_m, libmagma), magma_int_t, (magma_int_t, magma_range_t, magma_int_t, Cdouble, Cdouble, magma_int_t, magma_int_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magma_int_t}, magma_int_t, Ptr{magma_int_t}), ngpu, range, n, vl, vu, il, iu, d, e, Z, ldz, rwork, lrwork, iwork, liwork, info)
end

"""
    magma_ztrevc3(side, howmany, select, n, T, ldt, VL, ldvl, VR, ldvr, mm, mout, work, lwork, rwork, info)

------------------------------------------------------------ ztr routines
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_ztrevc3( magma_side_t side, magma_vec_t howmany, magma_int_t *select, magma_int_t n, magmaDoubleComplex *T, magma_int_t ldt, magmaDoubleComplex *VL, magma_int_t ldvl, magmaDoubleComplex *VR, magma_int_t ldvr, magma_int_t mm, magma_int_t *mout, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_ztrevc3(side, howmany, select, n, T, ldt, VL, ldvl, VR, ldvr, mm, mout, work, lwork, rwork, info)
    ccall((:magma_ztrevc3, libmagma), magma_int_t, (magma_side_t, magma_vec_t, Ptr{magma_int_t}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), side, howmany, select, n, T, ldt, VL, ldvl, VR, ldvr, mm, mout, work, lwork, rwork, info)
end

"""
    magma_ztrevc3_mt(side, howmany, select, n, T, ldt, VL, ldvl, VR, ldvr, mm, mout, work, lwork, rwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_ztrevc3_mt( magma_side_t side, magma_vec_t howmany, magma_int_t *select, magma_int_t n, magmaDoubleComplex *T, magma_int_t ldt, magmaDoubleComplex *VL, magma_int_t ldvl, magmaDoubleComplex *VR, magma_int_t ldvr, magma_int_t mm, magma_int_t *mout, magmaDoubleComplex *work, magma_int_t lwork, #ifdef MAGMA_COMPLEX double *rwork, #endif magma_int_t *info);
```
"""
function magma_ztrevc3_mt(side, howmany, select, n, T, ldt, VL, ldvl, VR, ldvr, mm, mout, work, lwork, rwork, info)
    ccall((:magma_ztrevc3_mt, libmagma), magma_int_t, (magma_side_t, magma_vec_t, Ptr{magma_int_t}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, Ptr{magma_int_t}), side, howmany, select, n, T, ldt, VL, ldvl, VR, ldvr, mm, mout, work, lwork, rwork, info)
end

"""
    magma_ztrsm_m(ngpu, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_ztrsm_m( magma_int_t ngpu, magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha, const magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb);
```
"""
function magma_ztrsm_m(ngpu, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
    ccall((:magma_ztrsm_m, libmagma), magma_int_t, (magma_int_t, magma_side_t, magma_uplo_t, magma_trans_t, magma_diag_t, magma_int_t, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t), ngpu, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
end

"""
    magma_ztrtri(uplo, diag, n, A, lda, info)


### Prototype
```c
magma_int_t magma_ztrtri( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);
```
"""
function magma_ztrtri(uplo, diag, n, A, lda, info)
    ccall((:magma_ztrtri, libmagma), magma_int_t, (magma_uplo_t, magma_diag_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), uplo, diag, n, A, lda, info)
end

"""
    magma_ztrtri_gpu(uplo, diag, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_ztrtri_gpu( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info);
```
"""
function magma_ztrtri_gpu(uplo, diag, n, dA, ldda, info)
    ccall((:magma_ztrtri_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_diag_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), uplo, diag, n, dA, ldda, info)
end

"""
    magma_zungbr(vect, m, n, k, A, lda, tau, work, lwork, info)

------------------------------------------------------------ zun routines
CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zungbr( magma_vect_t vect, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zungbr(vect, m, n, k, A, lda, tau, work, lwork, info)
    ccall((:magma_zungbr, libmagma), magma_int_t, (magma_vect_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), vect, m, n, k, A, lda, tau, work, lwork, info)
end

"""
    magma_zunghr(n, ilo, ihi, A, lda, tau, dT, nb, info)


### Prototype
```c
magma_int_t magma_zunghr( magma_int_t n, magma_int_t ilo, magma_int_t ihi, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zunghr(n, ilo, ihi, A, lda, tau, dT, nb, info)
    ccall((:magma_zunghr, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), n, ilo, ihi, A, lda, tau, dT, nb, info)
end

"""
    magma_zunghr_m(n, ilo, ihi, A, lda, tau, T, nb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunghr_m( magma_int_t n, magma_int_t ilo, magma_int_t ihi, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *T, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zunghr_m(n, ilo, ihi, A, lda, tau, T, nb, info)
    ccall((:magma_zunghr_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), n, ilo, ihi, A, lda, tau, T, nb, info)
end

"""
    magma_zunglq(m, n, k, A, lda, tau, dT, nb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunglq( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zunglq(m, n, k, A, lda, tau, dT, nb, info)
    ccall((:magma_zunglq, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, k, A, lda, tau, dT, nb, info)
end

"""
    magma_zungqr(m, n, k, A, lda, tau, dT, nb, info)


### Prototype
```c
magma_int_t magma_zungqr( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zungqr(m, n, k, A, lda, tau, dT, nb, info)
    ccall((:magma_zungqr, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, k, A, lda, tau, dT, nb, info)
end

"""
    magma_zungqr_gpu(m, n, k, dA, ldda, tau, dT, nb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zungqr_gpu( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zungqr_gpu(m, n, k, dA, ldda, tau, dT, nb, info)
    ccall((:magma_zungqr_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), m, n, k, dA, ldda, tau, dT, nb, info)
end

"""
    magma_zungqr_m(m, n, k, A, lda, tau, T, nb, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zungqr_m( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *T, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zungqr_m(m, n, k, A, lda, tau, T, nb, info)
    ccall((:magma_zungqr_m, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), m, n, k, A, lda, tau, T, nb, info)
end

"""
    magma_zungqr2(m, n, k, A, lda, tau, info)


### Prototype
```c
magma_int_t magma_zungqr2( magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magma_int_t *info);
```
"""
function magma_zungqr2(m, n, k, A, lda, tau, info)
    ccall((:magma_zungqr2, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magma_int_t}), m, n, k, A, lda, tau, info)
end

"""
    magma_zunmbr(vect, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)


### Prototype
```c
magma_int_t magma_zunmbr( magma_vect_t vect, magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmbr(vect, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmbr, libmagma), magma_int_t, (magma_vect_t, magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), vect, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmlq(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)


### Prototype
```c
magma_int_t magma_zunmlq( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmlq(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmlq, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmrq(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)


### Prototype
```c
magma_int_t magma_zunmrq( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmrq(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmrq, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmql(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)


### Prototype
```c
magma_int_t magma_zunmql( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmql(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmql, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmql2_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunmql2_gpu( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dC, magma_int_t lddc, const magmaDoubleComplex *wA, magma_int_t ldwa, magma_int_t *info);
```
"""
function magma_zunmql2_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info)
    ccall((:magma_zunmql2_gpu, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info)
end

"""
    magma_zunmqr(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)


### Prototype
```c
magma_int_t magma_zunmqr( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmqr(side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmqr, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmqr_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, hwork, lwork, dT, nb, info)


### Prototype
```c
magma_int_t magma_zunmqr_gpu( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magmaDoubleComplex const *tau, magmaDoubleComplex_ptr dC, magma_int_t lddc, magmaDoubleComplex *hwork, magma_int_t lwork, magmaDoubleComplex_ptr dT, magma_int_t nb, magma_int_t *info);
```
"""
function magma_zunmqr_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, hwork, lwork, dT, nb, info)
    ccall((:magma_zunmqr_gpu, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, dA, ldda, tau, dC, lddc, hwork, lwork, dT, nb, info)
end

"""
    magma_zunmqr2_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunmqr2_gpu( magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dC, magma_int_t lddc, const magmaDoubleComplex *wA, magma_int_t ldwa, magma_int_t *info);
```
"""
function magma_zunmqr2_gpu(side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info)
    ccall((:magma_zunmqr2_gpu, libmagma), magma_int_t, (magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, trans, m, n, k, dA, ldda, tau, dC, lddc, wA, ldwa, info)
end

"""
    magma_zunmqr_m(ngpu, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunmqr_m( magma_int_t ngpu, magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmqr_m(ngpu, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmqr_m, libmagma), magma_int_t, (magma_int_t, magma_side_t, magma_trans_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmtr(side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)


### Prototype
```c
magma_int_t magma_zunmtr( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmtr(side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmtr, libmagma), magma_int_t, (magma_side_t, magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_zunmtr_gpu(side, uplo, trans, m, n, dA, ldda, tau, dC, lddc, wA, ldwa, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunmtr_gpu( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex *tau, magmaDoubleComplex_ptr dC, magma_int_t lddc, const magmaDoubleComplex *wA, magma_int_t ldwa, magma_int_t *info);
```
"""
function magma_zunmtr_gpu(side, uplo, trans, m, n, dA, ldda, tau, dC, lddc, wA, ldwa, info)
    ccall((:magma_zunmtr_gpu, libmagma), magma_int_t, (magma_side_t, magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magmaDoubleComplex_ptr, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), side, uplo, trans, m, n, dA, ldda, tau, dC, lddc, wA, ldwa, info)
end

"""
    magma_zunmtr_m(ngpu, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zunmtr_m( magma_int_t ngpu, magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *C, magma_int_t ldc, magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);
```
"""
function magma_zunmtr_m(ngpu, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    ccall((:magma_zunmtr_m, libmagma), magma_int_t, (magma_int_t, magma_side_t, magma_uplo_t, magma_trans_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}), ngpu, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
end

"""
    magma_z_isnan(x)


### Prototype
```c
int magma_z_isnan( magmaDoubleComplex x );
```
"""
function magma_z_isnan(x)
    ccall((:magma_z_isnan, libmagma), Cint, (magmaDoubleComplex,), x)
end

"""
    magma_z_isinf(x)


### Prototype
```c
int magma_z_isinf( magmaDoubleComplex x );
```
"""
function magma_z_isinf(x)
    ccall((:magma_z_isinf, libmagma), Cint, (magmaDoubleComplex,), x)
end

"""
    magma_z_isnan_inf(x)


### Prototype
```c
int magma_z_isnan_inf( magmaDoubleComplex x );
```
"""
function magma_z_isnan_inf(x)
    ccall((:magma_z_isnan_inf, libmagma), Cint, (magmaDoubleComplex,), x)
end

"""
    magma_zmake_lwork(lwork)


### Prototype
```c
magmaDoubleComplex magma_zmake_lwork( magma_int_t lwork );
```
"""
function magma_zmake_lwork(lwork)
    ccall((:magma_zmake_lwork, libmagma), magmaDoubleComplex, (magma_int_t,), lwork)
end

"""
    magma_znan_inf(uplo, m, n, A, lda, cnt_nan, cnt_inf)


### Prototype
```c
magma_int_t magma_znan_inf( magma_uplo_t uplo, magma_int_t m, magma_int_t n, const magmaDoubleComplex *A, magma_int_t lda, magma_int_t *cnt_nan, magma_int_t *cnt_inf);
```
"""
function magma_znan_inf(uplo, m, n, A, lda, cnt_nan, cnt_inf)
    ccall((:magma_znan_inf, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, m, n, A, lda, cnt_nan, cnt_inf)
end

"""
    magma_znan_inf_gpu(uplo, m, n, dA, ldda, cnt_nan, cnt_inf, queue)


### Prototype
```c
magma_int_t magma_znan_inf_gpu( magma_uplo_t uplo, magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magma_int_t *cnt_nan, magma_int_t *cnt_inf, magma_queue_t queue);
```
"""
function magma_znan_inf_gpu(uplo, m, n, dA, ldda, cnt_nan, cnt_inf, queue)
    ccall((:magma_znan_inf_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_queue_t), uplo, m, n, dA, ldda, cnt_nan, cnt_inf, queue)
end

"""
    magma_zprint(m, n, A, lda)


### Prototype
```c
void magma_zprint( magma_int_t m, magma_int_t n, const magmaDoubleComplex *A, magma_int_t lda);
```
"""
function magma_zprint(m, n, A, lda)
    ccall((:magma_zprint, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t), m, n, A, lda)
end

"""
    magma_zprint_gpu(m, n, dA, ldda, queue)


### Prototype
```c
void magma_zprint_gpu( magma_int_t m, magma_int_t n, magmaDoubleComplex_const_ptr dA, magma_int_t ldda, magma_queue_t queue);
```
"""
function magma_zprint_gpu(m, n, dA, ldda, queue)
    ccall((:magma_zprint_gpu, libmagma), Cvoid, (magma_int_t, magma_int_t, magmaDoubleComplex_const_ptr, magma_int_t, magma_queue_t), m, n, dA, ldda, queue)
end

"""
    magma_zpanel_to_q(uplo, ib, A, lda, work)


### Prototype
```c
void magma_zpanel_to_q( magma_uplo_t uplo, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *work);
```
"""
function magma_zpanel_to_q(uplo, ib, A, lda, work)
    ccall((:magma_zpanel_to_q, libmagma), Cvoid, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}), uplo, ib, A, lda, work)
end

"""
    magma_zq_to_panel(uplo, ib, A, lda, work)


### Prototype
```c
void magma_zq_to_panel( magma_uplo_t uplo, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *work);
```
"""
function magma_zq_to_panel(uplo, ib, A, lda, work)
    ccall((:magma_zq_to_panel, libmagma), Cvoid, (magma_uplo_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}), uplo, ib, A, lda, work)
end

"""
    magmablas_zextract_diag_sqrt(m, n, dA, ldda, dD, incd, queue)

auxiliary routines for posv-irgmres  
### Prototype
```c
void magmablas_zextract_diag_sqrt( magma_int_t m, magma_int_t n, magmaDoubleComplex* dA, magma_int_t ldda, double* dD, magma_int_t incd, magma_queue_t queue);
```
"""
function magmablas_zextract_diag_sqrt(m, n, dA, ldda, dD, incd, queue)
    ccall((:magmablas_zextract_diag_sqrt, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{Cdouble}, magma_int_t, magma_queue_t), m, n, dA, ldda, dD, incd, queue)
end

"""
    magmablas_zscal_shift_hpd(uplo, n, dA, ldda, dD, incd, miu, cn, eps, queue)


### Prototype
```c
void magmablas_zscal_shift_hpd( magma_uplo_t uplo, int n, magmaDoubleComplex* dA, int ldda, double* dD, int incd, double miu, double cn, double eps, magma_queue_t queue);
```
"""
function magmablas_zscal_shift_hpd(uplo, n, dA, ldda, dD, incd, miu, cn, eps, queue)
    ccall((:magmablas_zscal_shift_hpd, libmagma), Cvoid, (magma_uplo_t, Cint, Ptr{magmaDoubleComplex}, Cint, Ptr{Cdouble}, Cint, Cdouble, Cdouble, Cdouble, magma_queue_t), uplo, n, dA, ldda, dD, incd, miu, cn, eps, queue)
end

"""
    magmablas_zdimv_invert(n, alpha, dD, incd, dx, incx, beta, dy, incy, queue)


### Prototype
```c
void magmablas_zdimv_invert( magma_int_t n, magmaDoubleComplex alpha, magmaDoubleComplex* dD, magma_int_t incd, magmaDoubleComplex* dx, magma_int_t incx, magmaDoubleComplex beta, magmaDoubleComplex* dy, magma_int_t incy, magma_queue_t queue);
```
"""
function magmablas_zdimv_invert(n, alpha, dD, incd, dx, incx, beta, dy, incy, queue)
    ccall((:magmablas_zdimv_invert, libmagma), Cvoid, (magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, magma_int_t, Ptr{magmaDoubleComplex}, magma_int_t, magmaDoubleComplex, Ptr{magmaDoubleComplex}, magma_int_t, magma_queue_t), n, alpha, dD, incd, dx, incx, beta, dy, incy, queue)
end

"""
    magma_zcgeqrsv_gpu(m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)

=============================================================================
MAGMA mixed precision function definitions

In alphabetical order of base name (ignoring precision).
### Prototype
```c
magma_int_t magma_zcgeqrsv_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_zcgeqrsv_gpu(m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)
    ccall((:magma_zcgeqrsv_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)
end

"""
    magma_zcgesv_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)


### Prototype
```c
magma_int_t magma_zcgesv_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaInt_ptr dipiv, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magmaDoubleComplex_ptr dworkd, magmaFloatComplex_ptr dworks, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_zcgesv_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
    ccall((:magma_zcgesv_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, Ptr{magma_int_t}, magmaInt_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaFloatComplex_ptr, Ptr{magma_int_t}, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
end

"""
    magma_zcgetrs_gpu(trans, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dSX, info)


### Prototype
```c
magma_int_t magma_zcgetrs_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaFloatComplex_ptr dA, magma_int_t ldda, magmaInt_ptr dipiv, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magmaFloatComplex_ptr dSX, magma_int_t *info);
```
"""
function magma_zcgetrs_gpu(trans, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dSX, info)
    ccall((:magma_zcgetrs_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaFloatComplex_ptr, magma_int_t, magmaInt_ptr, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaFloatComplex_ptr, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dSX, info)
end

"""
    magma_zchesv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, info)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_zchesv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magmaDoubleComplex_ptr dworkd, magmaFloatComplex_ptr dworks, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_zchesv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
    ccall((:magma_zchesv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaFloatComplex_ptr, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
end

"""
    magma_zcposv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, info)


### Prototype
```c
magma_int_t magma_zcposv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDoubleComplex_ptr dA, magma_int_t ldda, magmaDoubleComplex_ptr dB, magma_int_t lddb, magmaDoubleComplex_ptr dX, magma_int_t lddx, magmaDoubleComplex_ptr dworkd, magmaFloatComplex_ptr dworks, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_zcposv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
    ccall((:magma_zcposv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magma_int_t, magmaDoubleComplex_ptr, magmaFloatComplex_ptr, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
end

"""
    magma_init()

=============================================================================
initialization
### Prototype
```c
magma_int_t magma_init( void );
```
"""
function magma_init()
    ccall((:magma_init, libmagma), magma_int_t, ())
end

"""
    magma_finalize()


### Prototype
```c
magma_int_t magma_finalize( void );
```
"""
function magma_finalize()
    ccall((:magma_finalize, libmagma), magma_int_t, ())
end

"""
    magma_version(major, minor, micro)

=============================================================================
version information
### Prototype
```c
void magma_version( magma_int_t* major, magma_int_t* minor, magma_int_t* micro );
```
"""
function magma_version(major, minor, micro)
    ccall((:magma_version, libmagma), Cvoid, (Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{magma_int_t}), major, minor, micro)
end

# no prototype is found for this function at magma_auxiliary.h:42:6, please use with caution
"""
    magma_print_environment()


### Prototype
```c
void magma_print_environment();
```
"""
function magma_print_environment()
    ccall((:magma_print_environment, libmagma), Cvoid, ())
end

"""
    magma_wtime()

=============================================================================
timing
### Prototype
```c
real_Double_t magma_wtime( void );
```
"""
function magma_wtime()
    ccall((:magma_wtime, libmagma), real_Double_t, ())
end

"""
    magma_sync_wtime(queue)


### Prototype
```c
real_Double_t magma_sync_wtime( magma_queue_t queue );
```
"""
function magma_sync_wtime(queue)
    ccall((:magma_sync_wtime, libmagma), real_Double_t, (magma_queue_t,), queue)
end

"""
    magma_buildconnection_mgpu(gnode, ncmplx, ngpu)

magma GPU-complex PCIe connection
### Prototype
```c
magma_int_t magma_buildconnection_mgpu( magma_int_t gnode[MagmaMaxGPUs+2][MagmaMaxGPUs+2], magma_int_t *ncmplx, magma_int_t ngpu );
```
"""
function magma_buildconnection_mgpu(gnode, ncmplx, ngpu)
    ccall((:magma_buildconnection_mgpu, libmagma), magma_int_t, (Ptr{NTuple{10, magma_int_t}}, Ptr{magma_int_t}, magma_int_t), gnode, ncmplx, ngpu)
end

"""
    magma_indices_1D_bcyclic(nb, ngpu, dev, j0, j1, dj0, dj1)


### Prototype
```c
void magma_indices_1D_bcyclic( magma_int_t nb, magma_int_t ngpu, magma_int_t dev, magma_int_t j0, magma_int_t j1, magma_int_t* dj0, magma_int_t* dj1 );
```
"""
function magma_indices_1D_bcyclic(nb, ngpu, dev, j0, j1, dj0, dj1)
    ccall((:magma_indices_1D_bcyclic, libmagma), Cvoid, (magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), nb, ngpu, dev, j0, j1, dj0, dj1)
end

"""
    magma_swp2pswp(trans, n, ipiv, newipiv)


### Prototype
```c
void magma_swp2pswp( magma_trans_t trans, magma_int_t n, magma_int_t *ipiv, magma_int_t *newipiv );
```
"""
function magma_swp2pswp(trans, n, ipiv, newipiv)
    ccall((:magma_swp2pswp, libmagma), Cvoid, (magma_trans_t, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), trans, n, ipiv, newipiv)
end

# no prototype is found for this function at magma_auxiliary.h:75:13, please use with caution
"""
    magma_get_smlsize_divideconquer()

=============================================================================
get NB blocksize
### Prototype
```c
magma_int_t magma_get_smlsize_divideconquer();
```
"""
function magma_get_smlsize_divideconquer()
    ccall((:magma_get_smlsize_divideconquer, libmagma), magma_int_t, ())
end

"""
    magma_malloc(ptr_ptr, bytes)

=============================================================================
memory allocation
### Prototype
```c
magma_int_t magma_malloc( magma_ptr *ptr_ptr, size_t bytes );
```
"""
function magma_malloc(ptr_ptr, bytes)
    ccall((:magma_malloc, libmagma), magma_int_t, (Ptr{magma_ptr}, Cint), ptr_ptr, bytes)
end

"""
    magma_malloc_cpu(ptr_ptr, bytes)


### Prototype
```c
magma_int_t magma_malloc_cpu( void **ptr_ptr, size_t bytes );
```
"""
function magma_malloc_cpu(ptr_ptr, bytes)
    ccall((:magma_malloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{Cvoid}}, Cint), ptr_ptr, bytes)
end

"""
    magma_malloc_pinned(ptr_ptr, bytes)


### Prototype
```c
magma_int_t magma_malloc_pinned( void **ptr_ptr, size_t bytes );
```
"""
function magma_malloc_pinned(ptr_ptr, bytes)
    ccall((:magma_malloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{Cvoid}}, Cint), ptr_ptr, bytes)
end

"""
    magma_free_cpu(ptr)


### Prototype
```c
magma_int_t magma_free_cpu( void *ptr );
```
"""
function magma_free_cpu(ptr)
    ccall((:magma_free_cpu, libmagma), magma_int_t, (Ptr{Cvoid},), ptr)
end

"""
    magma_mem_info(freeMem, totalMem)

returns memory info (basically a wrapper around cudaMemGetInfo
### Prototype
```c
magma_int_t magma_mem_info(size_t* freeMem, size_t* totalMem);
```
"""
function magma_mem_info(freeMem, totalMem)
    ccall((:magma_mem_info, libmagma), magma_int_t, (Ptr{Cint}, Ptr{Cint}), freeMem, totalMem)
end

"""
    magma_memset(ptr, value, count)

wrapper around cudaMemset
### Prototype
```c
magma_int_t magma_memset(void * ptr, int value, size_t count);
```
"""
function magma_memset(ptr, value, count)
    ccall((:magma_memset, libmagma), magma_int_t, (Ptr{Cvoid}, Cint, Cint), ptr, value, count)
end

"""
    magma_memset_async(ptr, value, count, queue)

wrapper around cudaMemsetAsync
### Prototype
```c
magma_int_t magma_memset_async(void * ptr, int value, size_t count, magma_queue_t queue);
```
"""
function magma_memset_async(ptr, value, count, queue)
    ccall((:magma_memset_async, libmagma), magma_int_t, (Ptr{Cvoid}, Cint, Cint, magma_queue_t), ptr, value, count, queue)
end

"""
    magma_imalloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
### Prototype
```c
static inline magma_int_t magma_imalloc( magmaInt_ptr *ptr_ptr, size_t n );
```
"""
function magma_imalloc(ptr_ptr, n)
    ccall((:magma_imalloc, libmagma), magma_int_t, (Ptr{magmaInt_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_index_malloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for magma_index_t arrays. Allocates n*sizeof(magma_index_t) bytes.
### Prototype
```c
static inline magma_int_t magma_index_malloc( magmaIndex_ptr *ptr_ptr, size_t n );
```
"""
function magma_index_malloc(ptr_ptr, n)
    ccall((:magma_index_malloc, libmagma), magma_int_t, (Ptr{magmaIndex_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_uindex_malloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for magma_uindex_t arrays. Allocates n*sizeof(magma_uindex_t) bytes.
### Prototype
```c
static inline magma_int_t magma_uindex_malloc( magmaUIndex_ptr *ptr_ptr, size_t n );
```
"""
function magma_uindex_malloc(ptr_ptr, n)
    ccall((:magma_uindex_malloc, libmagma), magma_int_t, (Ptr{magmaUIndex_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_smalloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for float arrays. Allocates n*sizeof(float) bytes.
### Prototype
```c
static inline magma_int_t magma_smalloc( magmaFloat_ptr *ptr_ptr, size_t n );
```
"""
function magma_smalloc(ptr_ptr, n)
    ccall((:magma_smalloc, libmagma), magma_int_t, (Ptr{magmaFloat_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_dmalloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for double arrays. Allocates n*sizeof(double) bytes.
### Prototype
```c
static inline magma_int_t magma_dmalloc( magmaDouble_ptr *ptr_ptr, size_t n );
```
"""
function magma_dmalloc(ptr_ptr, n)
    ccall((:magma_dmalloc, libmagma), magma_int_t, (Ptr{magmaDouble_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_cmalloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for magmaFloatComplex arrays. Allocates n*sizeof(magmaFloatComplex) bytes.
### Prototype
```c
static inline magma_int_t magma_cmalloc( magmaFloatComplex_ptr *ptr_ptr, size_t n );
```
"""
function magma_cmalloc(ptr_ptr, n)
    ccall((:magma_cmalloc, libmagma), magma_int_t, (Ptr{magmaFloatComplex_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_zmalloc(ptr_ptr, n)

Type-safe version of magma_malloc(), for magmaDoubleComplex arrays. Allocates n*sizeof(magmaDoubleComplex) bytes.
### Prototype
```c
static inline magma_int_t magma_zmalloc( magmaDoubleComplex_ptr *ptr_ptr, size_t n );
```
"""
function magma_zmalloc(ptr_ptr, n)
    ccall((:magma_zmalloc, libmagma), magma_int_t, (Ptr{magmaDoubleComplex_ptr}, Cint), ptr_ptr, n)
end

"""
    magma_imalloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
### Prototype
```c
static inline magma_int_t magma_imalloc_cpu( magma_int_t **ptr_ptr, size_t n );
```
"""
function magma_imalloc_cpu(ptr_ptr, n)
    ccall((:magma_imalloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{magma_int_t}}, Cint), ptr_ptr, n)
end

"""
    magma_index_malloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for magma_index_t arrays. Allocates n*sizeof(magma_index_t) bytes.
### Prototype
```c
static inline magma_int_t magma_index_malloc_cpu( magma_index_t **ptr_ptr, size_t n );
```
"""
function magma_index_malloc_cpu(ptr_ptr, n)
    ccall((:magma_index_malloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{magma_index_t}}, Cint), ptr_ptr, n)
end

"""
    magma_uindex_malloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for magma_uindex_t arrays. Allocates n*sizeof(magma_uindex_t) bytes.
### Prototype
```c
static inline magma_int_t magma_uindex_malloc_cpu( magma_uindex_t **ptr_ptr, size_t n );
```
"""
function magma_uindex_malloc_cpu(ptr_ptr, n)
    ccall((:magma_uindex_malloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{magma_uindex_t}}, Cint), ptr_ptr, n)
end

"""
    magma_smalloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for float arrays. Allocates n*sizeof(float) bytes.
### Prototype
```c
static inline magma_int_t magma_smalloc_cpu( float **ptr_ptr, size_t n );
```
"""
function magma_smalloc_cpu(ptr_ptr, n)
    ccall((:magma_smalloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{Cfloat}}, Cint), ptr_ptr, n)
end

"""
    magma_dmalloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for double arrays. Allocates n*sizeof(double) bytes.
### Prototype
```c
static inline magma_int_t magma_dmalloc_cpu( double **ptr_ptr, size_t n );
```
"""
function magma_dmalloc_cpu(ptr_ptr, n)
    ccall((:magma_dmalloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{Cdouble}}, Cint), ptr_ptr, n)
end

"""
    magma_cmalloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for magmaFloatComplex arrays. Allocates n*sizeof(magmaFloatComplex) bytes.
### Prototype
```c
static inline magma_int_t magma_cmalloc_cpu( magmaFloatComplex **ptr_ptr, size_t n );
```
"""
function magma_cmalloc_cpu(ptr_ptr, n)
    ccall((:magma_cmalloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{magmaFloatComplex}}, Cint), ptr_ptr, n)
end

"""
    magma_zmalloc_cpu(ptr_ptr, n)

Type-safe version of magma_malloc_cpu(), for magmaDoubleComplex arrays. Allocates n*sizeof(magmaDoubleComplex) bytes.
### Prototype
```c
static inline magma_int_t magma_zmalloc_cpu( magmaDoubleComplex **ptr_ptr, size_t n );
```
"""
function magma_zmalloc_cpu(ptr_ptr, n)
    ccall((:magma_zmalloc_cpu, libmagma), magma_int_t, (Ptr{Ptr{magmaDoubleComplex}}, Cint), ptr_ptr, n)
end

"""
    magma_imalloc_pinned(ptr_ptr, n)

Type-safe version of magma_malloc_pinned(), for magma_int_t arrays. Allocates n*sizeof(magma_int_t) bytes.
### Prototype
```c
static inline magma_int_t magma_imalloc_pinned( magma_int_t **ptr_ptr, size_t n );
```
"""
function magma_imalloc_pinned(ptr_ptr, n)
    ccall((:magma_imalloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{magma_int_t}}, Cint), ptr_ptr, n)
end

"""
    magma_index_malloc_pinned(ptr_ptr, n)

Type-safe version of magma_malloc_pinned(), for magma_index_t arrays. Allocates n*sizeof(magma_index_t) bytes.
### Prototype
```c
static inline magma_int_t magma_index_malloc_pinned( magma_index_t **ptr_ptr, size_t n );
```
"""
function magma_index_malloc_pinned(ptr_ptr, n)
    ccall((:magma_index_malloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{magma_index_t}}, Cint), ptr_ptr, n)
end

"""
    magma_smalloc_pinned(ptr_ptr, n)

Type-safe version of magma_malloc_pinned(), for float arrays. Allocates n*sizeof(float) bytes.
### Prototype
```c
static inline magma_int_t magma_smalloc_pinned( float **ptr_ptr, size_t n );
```
"""
function magma_smalloc_pinned(ptr_ptr, n)
    ccall((:magma_smalloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{Cfloat}}, Cint), ptr_ptr, n)
end

"""
    magma_dmalloc_pinned(ptr_ptr, n)

Type-safe version of magma_malloc_pinned(), for double arrays. Allocates n*sizeof(double) bytes.
### Prototype
```c
static inline magma_int_t magma_dmalloc_pinned( double **ptr_ptr, size_t n );
```
"""
function magma_dmalloc_pinned(ptr_ptr, n)
    ccall((:magma_dmalloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{Cdouble}}, Cint), ptr_ptr, n)
end

"""
    magma_cmalloc_pinned(ptr_ptr, n)

Type-safe version of magma_malloc_pinned(), for magmaFloatComplex arrays. Allocates n*sizeof(magmaFloatComplex) bytes.
### Prototype
```c
static inline magma_int_t magma_cmalloc_pinned( magmaFloatComplex **ptr_ptr, size_t n );
```
"""
function magma_cmalloc_pinned(ptr_ptr, n)
    ccall((:magma_cmalloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{magmaFloatComplex}}, Cint), ptr_ptr, n)
end

"""
    magma_zmalloc_pinned(ptr_ptr, n)

Type-safe version of magma_malloc_pinned(), for magmaDoubleComplex arrays. Allocates n*sizeof(magmaDoubleComplex) bytes.
### Prototype
```c
static inline magma_int_t magma_zmalloc_pinned( magmaDoubleComplex **ptr_ptr, size_t n );
```
"""
function magma_zmalloc_pinned(ptr_ptr, n)
    ccall((:magma_zmalloc_pinned, libmagma), magma_int_t, (Ptr{Ptr{magmaDoubleComplex}}, Cint), ptr_ptr, n)
end

"""
    magma_is_devptr(ptr)

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_is_devptr( const void* ptr );
```
"""
function magma_is_devptr(ptr)
    ccall((:magma_is_devptr, libmagma), magma_int_t, (Ptr{Cvoid},), ptr)
end

"""
    magma_num_gpus()

=============================================================================
device support
### Prototype
```c
magma_int_t magma_num_gpus( void );
```
"""
function magma_num_gpus()
    ccall((:magma_num_gpus, libmagma), magma_int_t, ())
end

# no prototype is found for this function at magma_auxiliary.h:220:1, please use with caution
"""
    magma_getdevice_arch()

CUDA MAGMA only
### Prototype
```c
magma_int_t magma_getdevice_arch();
```
"""
function magma_getdevice_arch()
    ccall((:magma_getdevice_arch, libmagma), magma_int_t, ())
end

"""
    magma_getdevices(devices, size, num_dev)

magma_int_t magma_getdevice_arch( magma_int_t dev or queue );   todo: new 
### Prototype
```c
void magma_getdevices( magma_device_t* devices, magma_int_t size, magma_int_t* num_dev );
```
"""
function magma_getdevices(devices, size, num_dev)
    ccall((:magma_getdevices, libmagma), Cvoid, (Ptr{magma_device_t}, magma_int_t, Ptr{magma_int_t}), devices, size, num_dev)
end

"""
    magma_getdevice(dev)


### Prototype
```c
void magma_getdevice( magma_device_t* dev );
```
"""
function magma_getdevice(dev)
    ccall((:magma_getdevice, libmagma), Cvoid, (Ptr{magma_device_t},), dev)
end

"""
    magma_setdevice(dev)


### Prototype
```c
void magma_setdevice( magma_device_t dev );
```
"""
function magma_setdevice(dev)
    ccall((:magma_setdevice, libmagma), Cvoid, (magma_device_t,), dev)
end

"""
    magma_mem_size(queue)


### Prototype
```c
size_t magma_mem_size( magma_queue_t queue );
```
"""
function magma_mem_size(queue)
    ccall((:magma_mem_size, libmagma), Cint, (magma_queue_t,), queue)
end

# no prototype is found for this function at magma_auxiliary.h:239:1, please use with caution
"""
    magma_getdevice_multiprocessor_count()


### Prototype
```c
magma_int_t magma_getdevice_multiprocessor_count();
```
"""
function magma_getdevice_multiprocessor_count()
    ccall((:magma_getdevice_multiprocessor_count, libmagma), magma_int_t, ())
end

# no prototype is found for this function at magma_auxiliary.h:242:1, please use with caution
"""
    magma_getdevice_shmem_block()


### Prototype
```c
size_t magma_getdevice_shmem_block();
```
"""
function magma_getdevice_shmem_block()
    ccall((:magma_getdevice_shmem_block, libmagma), Cint, ())
end

# no prototype is found for this function at magma_auxiliary.h:245:1, please use with caution
"""
    magma_getdevice_shmem_multiprocessor()


### Prototype
```c
size_t magma_getdevice_shmem_multiprocessor();
```
"""
function magma_getdevice_shmem_multiprocessor()
    ccall((:magma_getdevice_shmem_multiprocessor, libmagma), Cint, ())
end

"""
    magma_queue_get_device(queue)


### Prototype
```c
magma_int_t magma_queue_get_device( magma_queue_t queue );
```
"""
function magma_queue_get_device(queue)
    ccall((:magma_queue_get_device, libmagma), magma_int_t, (magma_queue_t,), queue)
end

"""
    magma_event_create(event_ptr)

=============================================================================
event support
### Prototype
```c
void magma_event_create( magma_event_t* event_ptr );
```
"""
function magma_event_create(event_ptr)
    ccall((:magma_event_create, libmagma), Cvoid, (Ptr{magma_event_t},), event_ptr)
end

"""
    magma_event_create_untimed(event_ptr)


### Prototype
```c
void magma_event_create_untimed( magma_event_t* event_ptr );
```
"""
function magma_event_create_untimed(event_ptr)
    ccall((:magma_event_create_untimed, libmagma), Cvoid, (Ptr{magma_event_t},), event_ptr)
end

"""
    magma_event_destroy(event)


### Prototype
```c
void magma_event_destroy( magma_event_t event );
```
"""
function magma_event_destroy(event)
    ccall((:magma_event_destroy, libmagma), Cvoid, (magma_event_t,), event)
end

"""
    magma_event_record(event, queue)


### Prototype
```c
void magma_event_record( magma_event_t event, magma_queue_t queue );
```
"""
function magma_event_record(event, queue)
    ccall((:magma_event_record, libmagma), Cvoid, (magma_event_t, magma_queue_t), event, queue)
end

"""
    magma_event_query(event)


### Prototype
```c
void magma_event_query( magma_event_t event );
```
"""
function magma_event_query(event)
    ccall((:magma_event_query, libmagma), Cvoid, (magma_event_t,), event)
end

"""
    magma_event_sync(event)


### Prototype
```c
void magma_event_sync( magma_event_t event );
```
"""
function magma_event_sync(event)
    ccall((:magma_event_sync, libmagma), Cvoid, (magma_event_t,), event)
end

"""
    magma_queue_wait_event(queue, event)


### Prototype
```c
void magma_queue_wait_event( magma_queue_t queue, magma_event_t event );
```
"""
function magma_queue_wait_event(queue, event)
    ccall((:magma_queue_wait_event, libmagma), Cvoid, (magma_queue_t, magma_event_t), queue, event)
end

"""
    magma_xerbla(name, info)

=============================================================================
error handler
### Prototype
```c
void magma_xerbla( const char *name, magma_int_t info );
```
"""
function magma_xerbla(name, info)
    ccall((:magma_xerbla, libmagma), Cvoid, (Ptr{Cchar}, magma_int_t), name, info)
end

"""
    magma_strerror(error)


### Prototype
```c
const char* magma_strerror( magma_int_t error );
```
"""
function magma_strerror(error)
    ccall((:magma_strerror, libmagma), Ptr{Cchar}, (magma_int_t,), error)
end

"""
    magma_strlcpy(dst, src, size)

=============================================================================
string functions
### Prototype
```c
size_t magma_strlcpy( char *dst, const char *src, size_t size );
```
"""
function magma_strlcpy(dst, src, size)
    ccall((:magma_strlcpy, libmagma), Cint, (Ptr{Cchar}, Ptr{Cchar}, Cint), dst, src, size)
end

"""
    magma_ceildiv(x, y)

For integers x >= 0, y > 0, returns ceil( x/y ).
For x == 0, this is 0.
@ingroup magma_ceildiv
### Prototype
```c
static inline magma_int_t magma_ceildiv( magma_int_t x, magma_int_t y );
```
"""
function magma_ceildiv(x, y)
    ccall((:magma_ceildiv, libmagma), magma_int_t, (magma_int_t, magma_int_t), x, y)
end

"""
    magma_roundup(x, y)

For integers x >= 0, y > 0, returns x rounded up to multiple of y.
That is, ceil(x/y)*y.
For x == 0, this is 0.
This implementation does not assume y is a power of 2.
@ingroup magma_ceildiv
### Prototype
```c
static inline magma_int_t magma_roundup( magma_int_t x, magma_int_t y );
```
"""
function magma_roundup(x, y)
    ccall((:magma_roundup, libmagma), magma_int_t, (magma_int_t, magma_int_t), x, y)
end

"""
    magma_ssqrt(x)

@return Square root of x. @ingroup magma_sqrt
### Prototype
```c
static inline float magma_ssqrt( float x );
```
"""
function magma_ssqrt(x)
    ccall((:magma_ssqrt, libmagma), Cfloat, (Cfloat,), x)
end

"""
    magma_dsqrt(x)

@return Square root of x. @ingroup magma_sqrt
### Prototype
```c
static inline double magma_dsqrt( double x );
```
"""
function magma_dsqrt(x)
    ccall((:magma_dsqrt, libmagma), Cdouble, (Cdouble,), x)
end

"""
    magma_csqrt(x)

@return Complex square root of x. @ingroup magma_sqrt
### Prototype
```c
magmaFloatComplex magma_csqrt( magmaFloatComplex x );
```
"""
function magma_csqrt(x)
    ccall((:magma_csqrt, libmagma), magmaFloatComplex, (magmaFloatComplex,), x)
end

"""
    magma_zsqrt(x)

@return Complex square root of x. @ingroup magma_sqrt
### Prototype
```c
magmaDoubleComplex magma_zsqrt( magmaDoubleComplex x );
```
"""
function magma_zsqrt(x)
    ccall((:magma_zsqrt, libmagma), magmaDoubleComplex, (magmaDoubleComplex,), x)
end

"""
    magma_dhgesv_iteref_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)

Half precision iterative refinement routines 
### Prototype
```c
magma_int_t magma_dhgesv_iteref_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaInt_ptr dipiv, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magmaDouble_ptr dworkd, magmaFloat_ptr dworks, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_dhgesv_iteref_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
    ccall((:magma_dhgesv_iteref_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, Ptr{magma_int_t}, magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magmaFloat_ptr, Ptr{magma_int_t}, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
end

"""
    magma_dsgesv_iteref_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)


### Prototype
```c
magma_int_t magma_dsgesv_iteref_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaInt_ptr dipiv, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magmaDouble_ptr dworkd, magmaFloat_ptr dworks, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_dsgesv_iteref_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
    ccall((:magma_dsgesv_iteref_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, Ptr{magma_int_t}, magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magmaFloat_ptr, Ptr{magma_int_t}, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, iter, info)
end

"""
    magma_dxgesv_gmres_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, facto_type, solver_type, iter, info, facto_time)


### Prototype
```c
magma_int_t magma_dxgesv_gmres_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magmaInt_ptr dipiv, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magmaDouble_ptr dworkd, magmaFloat_ptr dworks, magma_refinement_t facto_type, magma_refinement_t solver_type, magma_int_t *iter, magma_int_t *info, real_Double_t *facto_time);
```
"""
function magma_dxgesv_gmres_gpu(trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, facto_type, solver_type, iter, info, facto_time)
    ccall((:magma_dxgesv_gmres_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, Ptr{magma_int_t}, magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magmaFloat_ptr, magma_refinement_t, magma_refinement_t, Ptr{magma_int_t}, Ptr{magma_int_t}, Ptr{real_Double_t}), trans, n, nrhs, dA, ldda, ipiv, dipiv, dB, lddb, dX, lddx, dworkd, dworks, facto_type, solver_type, iter, info, facto_time)
end

"""
    magma_dfgmres_plu_gpu(trans, n, nrhs, dA, ldda, dLU_sprec, lddlusp, dLU_dprec, lddludp, ipiv, dipiv, dB, lddb, dX, lddx, dSX, maxiter, restrt, maxiter_inner, restrt_inner, userinitguess, tol, innertol, rnorm0, niters, solver_type, algoname, is_inner, queue)


### Prototype
```c
magma_int_t magma_dfgmres_plu_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magmaFloat_ptr dLU_sprec, magma_int_t lddlusp, magmaDouble_ptr dLU_dprec, magma_int_t lddludp, magmaInt_ptr ipiv, magmaInt_ptr dipiv, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magmaFloat_ptr dSX, magma_int_t maxiter, magma_int_t restrt, magma_int_t maxiter_inner, magma_int_t restrt_inner, magma_int_t userinitguess, double tol, double innertol, double *rnorm0, magma_int_t *niters, magma_refinement_t solver_type, char *algoname, magma_int_t is_inner, magma_queue_t queue);
```
"""
function magma_dfgmres_plu_gpu(trans, n, nrhs, dA, ldda, dLU_sprec, lddlusp, dLU_dprec, lddludp, ipiv, dipiv, dB, lddb, dX, lddx, dSX, maxiter, restrt, maxiter_inner, restrt_inner, userinitguess, tol, innertol, rnorm0, niters, solver_type, algoname, is_inner, queue)
    ccall((:magma_dfgmres_plu_gpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magmaFloat_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaInt_ptr, magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaFloat_ptr, magma_int_t, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{magma_int_t}, magma_refinement_t, Ptr{Cchar}, magma_int_t, magma_queue_t), trans, n, nrhs, dA, ldda, dLU_sprec, lddlusp, dLU_dprec, lddludp, ipiv, dipiv, dB, lddb, dX, lddx, dSX, maxiter, restrt, maxiter_inner, restrt_inner, userinitguess, tol, innertol, rnorm0, niters, solver_type, algoname, is_inner, queue)
end

"""
    magma_dsgelatrs_cpu(trans, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dSX, info)


### Prototype
```c
magma_int_t magma_dsgelatrs_cpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs, magmaFloat_ptr dA, magma_int_t ldda, magmaInt_ptr dipiv, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magmaFloat_ptr dSX, magma_int_t *info);
```
"""
function magma_dsgelatrs_cpu(trans, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dSX, info)
    ccall((:magma_dsgelatrs_cpu, libmagma), magma_int_t, (magma_trans_t, magma_int_t, magma_int_t, magmaFloat_ptr, magma_int_t, magmaInt_ptr, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaFloat_ptr, Ptr{magma_int_t}), trans, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dSX, info)
end

"""
    magma_hgetrf_gpu(m, n, dA, ldda, ipiv, info)

Half precision LU factorizations routines 
### Prototype
```c
magma_int_t magma_hgetrf_gpu( magma_int_t m, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info );
```
"""
function magma_hgetrf_gpu(m, n, dA, ldda, ipiv, info)
    ccall((:magma_hgetrf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaFloat_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, dA, ldda, ipiv, info)
end

"""
    magma_htgetrf_gpu(m, n, dA, ldda, ipiv, info)


### Prototype
```c
magma_int_t magma_htgetrf_gpu( magma_int_t m, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info );
```
"""
function magma_htgetrf_gpu(m, n, dA, ldda, ipiv, info)
    ccall((:magma_htgetrf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaFloat_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), m, n, dA, ldda, ipiv, info)
end

"""
    magma_xhsgetrf_gpu(m, n, dA, ldda, ipiv, info, enable_tc, mp_algo_type)


### Prototype
```c
magma_int_t magma_xhsgetrf_gpu( magma_int_t m, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info, magma_mp_type_t enable_tc, magma_mp_type_t mp_algo_type);
```
"""
function magma_xhsgetrf_gpu(m, n, dA, ldda, ipiv, info, enable_tc, mp_algo_type)
    ccall((:magma_xhsgetrf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaFloat_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_mp_type_t, magma_mp_type_t), m, n, dA, ldda, ipiv, info, enable_tc, mp_algo_type)
end

"""
    magma_xshgetrf_gpu(m, n, dA, ldda, ipiv, info, enable_tc, mp_algo_type)


### Prototype
```c
magma_int_t magma_xshgetrf_gpu( magma_int_t m, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info, magma_mp_type_t enable_tc, magma_mp_type_t mp_algo_type);
```
"""
function magma_xshgetrf_gpu(m, n, dA, ldda, ipiv, info, enable_tc, mp_algo_type)
    ccall((:magma_xshgetrf_gpu, libmagma), magma_int_t, (magma_int_t, magma_int_t, magmaFloat_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}, magma_mp_type_t, magma_mp_type_t), m, n, dA, ldda, ipiv, info, enable_tc, mp_algo_type)
end

"""
    magma_get_hgetrf_nb(m, n)


### Prototype
```c
magma_int_t magma_get_hgetrf_nb( magma_int_t m, magma_int_t n );
```
"""
function magma_get_hgetrf_nb(m, n)
    ccall((:magma_get_hgetrf_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t), m, n)
end

"""
    magma_get_xgetrf_nb(m, n, prev_nb, enable_tc, mp_algo_type)


### Prototype
```c
magma_int_t magma_get_xgetrf_nb( magma_int_t m, magma_int_t n, magma_int_t prev_nb, magma_mp_type_t enable_tc, magma_mp_type_t mp_algo_type);
```
"""
function magma_get_xgetrf_nb(m, n, prev_nb, enable_tc, mp_algo_type)
    ccall((:magma_get_xgetrf_nb, libmagma), magma_int_t, (magma_int_t, magma_int_t, magma_int_t, magma_mp_type_t, magma_mp_type_t), m, n, prev_nb, enable_tc, mp_algo_type)
end

"""
    magma_dshposv_gpu_expert(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, mode, use_gmres, preprocess, cn, theta, info)

Cholesky-based solvers with FP16 capability 
### Prototype
```c
magma_int_t magma_dshposv_gpu_expert( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magmaDouble_ptr dworkd, magmaFloat_ptr dworks, magma_int_t *iter, magma_mode_t mode, magma_int_t use_gmres, magma_int_t preprocess, float cn, float theta, magma_int_t *info);
```
"""
function magma_dshposv_gpu_expert(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, mode, use_gmres, preprocess, cn, theta, info)
    ccall((:magma_dshposv_gpu_expert, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magmaFloat_ptr, Ptr{magma_int_t}, magma_mode_t, magma_int_t, magma_int_t, Cfloat, Cfloat, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dworkd, dworks, iter, mode, use_gmres, preprocess, cn, theta, info)
end

"""
    magma_dshposv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)


### Prototype
```c
magma_int_t magma_dshposv_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_dshposv_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)
    ccall((:magma_dshposv_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)
end

"""
    magma_dshposv_native(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)


### Prototype
```c
magma_int_t magma_dshposv_native( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, magmaDouble_ptr dA, magma_int_t ldda, magmaDouble_ptr dB, magma_int_t lddb, magmaDouble_ptr dX, magma_int_t lddx, magma_int_t *iter, magma_int_t *info);
```
"""
function magma_dshposv_native(uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)
    ccall((:magma_dshposv_native, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, magmaDouble_ptr, magma_int_t, Ptr{magma_int_t}, Ptr{magma_int_t}), uplo, n, nrhs, dA, ldda, dB, lddb, dX, lddx, iter, info)
end

"""
    magma_shpotrf_gpu(uplo, n, dA, ldda, info)

Cholesky factorizations routines with FP16 
### Prototype
```c
magma_int_t magma_shpotrf_gpu( magma_uplo_t uplo, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magma_shpotrf_gpu(uplo, n, dA, ldda, info)
    ccall((:magma_shpotrf_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaFloat_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_shpotrf_native(uplo, n, dA, ldda, info)


### Prototype
```c
magma_int_t magma_shpotrf_native( magma_uplo_t uplo, magma_int_t n, magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *info );
```
"""
function magma_shpotrf_native(uplo, n, dA, ldda, info)
    ccall((:magma_shpotrf_native, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magmaFloat_ptr, magma_int_t, Ptr{magma_int_t}), uplo, n, dA, ldda, info)
end

"""
    magma_dfgmres_spd_gpu(uplo, n, nrhs, dA, ldda, dL, lddl, dD, dB, lddb, dX, lddx, dSX, maxiter, restrt, maxiter_inner, restrt_inner, tol, innertol, rnorm0, niters, is_inner, is_preprocessed, miu, queue)


### Prototype
```c
magma_int_t magma_dfgmres_spd_gpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double *dA, magma_int_t ldda, float *dL, magma_int_t lddl, float* dD, double *dB, magma_int_t lddb, double *dX, magma_int_t lddx, float *dSX, magma_int_t maxiter, magma_int_t restrt, magma_int_t maxiter_inner, magma_int_t restrt_inner, double tol, double innertol, double *rnorm0, magma_int_t *niters, magma_int_t is_inner, magma_int_t is_preprocessed, float miu, magma_queue_t queue);
```
"""
function magma_dfgmres_spd_gpu(uplo, n, nrhs, dA, ldda, dL, lddl, dD, dB, lddb, dX, lddx, dSX, maxiter, restrt, maxiter_inner, restrt_inner, tol, innertol, rnorm0, niters, is_inner, is_preprocessed, miu, queue)
    ccall((:magma_dfgmres_spd_gpu, libmagma), magma_int_t, (magma_uplo_t, magma_int_t, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{Cfloat}, magma_int_t, Ptr{Cfloat}, Ptr{Cdouble}, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{Cfloat}, magma_int_t, magma_int_t, magma_int_t, magma_int_t, Cdouble, Cdouble, Ptr{Cdouble}, Ptr{magma_int_t}, magma_int_t, magma_int_t, Cfloat, magma_queue_t), uplo, n, nrhs, dA, ldda, dL, lddl, dD, dB, lddb, dX, lddx, dSX, maxiter, restrt, maxiter_inner, restrt_inner, tol, innertol, rnorm0, niters, is_inner, is_preprocessed, miu, queue)
end

"""
    magmablas_convert_dp2hp(m, n, dA, ldda, dB, lddb, queue)

Half precision conversion routines 
### Prototype
```c
void magmablas_convert_dp2hp( magma_int_t m, magma_int_t n, const double *dA, magma_int_t ldda, magmaHalf *dB, magma_int_t lddb, magma_queue_t queue );
```
"""
function magmablas_convert_dp2hp(m, n, dA, ldda, dB, lddb, queue)
    ccall((:magmablas_convert_dp2hp, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Cdouble}, magma_int_t, Ptr{magmaHalf}, magma_int_t, magma_queue_t), m, n, dA, ldda, dB, lddb, queue)
end

"""
    magmablas_convert_hp2dp(m, n, dA, ldda, dB, lddb, queue)


### Prototype
```c
void magmablas_convert_hp2dp( magma_int_t m, magma_int_t n, const magmaHalf *dA, magma_int_t ldda, double *dB, magma_int_t lddb, magma_queue_t queue );
```
"""
function magmablas_convert_hp2dp(m, n, dA, ldda, dB, lddb, queue)
    ccall((:magmablas_convert_hp2dp, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaHalf}, magma_int_t, Ptr{Cdouble}, magma_int_t, magma_queue_t), m, n, dA, ldda, dB, lddb, queue)
end

"""
    magmablas_convert_hp2sp(m, n, dA, ldda, dB, lddb, queue)


### Prototype
```c
void magmablas_convert_hp2sp( magma_int_t m, magma_int_t n, const magmaHalf *dA, magma_int_t ldda, float *dB, magma_int_t lddb, magma_queue_t queue );
```
"""
function magmablas_convert_hp2sp(m, n, dA, ldda, dB, lddb, queue)
    ccall((:magmablas_convert_hp2sp, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{magmaHalf}, magma_int_t, Ptr{Cfloat}, magma_int_t, magma_queue_t), m, n, dA, ldda, dB, lddb, queue)
end

"""
    magmablas_convert_sp2hp(m, n, dA, ldda, dB, lddb, queue)


### Prototype
```c
void magmablas_convert_sp2hp( magma_int_t m, magma_int_t n, const float *dA, magma_int_t ldda, magmaHalf *dB, magma_int_t lddb, magma_queue_t queue );
```
"""
function magmablas_convert_sp2hp(m, n, dA, ldda, dB, lddb, queue)
    ccall((:magmablas_convert_sp2hp, libmagma), Cvoid, (magma_int_t, magma_int_t, Ptr{Cfloat}, magma_int_t, Ptr{magmaHalf}, magma_int_t, magma_queue_t), m, n, dA, ldda, dB, lddb, queue)
end

"""
    magmablas_hlaswp(n, dAT, ldda, k1, k2, ipiv, inci, queue)


### Prototype
```c
void magmablas_hlaswp( magma_int_t n, magmaHalf *dAT, magma_int_t ldda, magma_int_t k1, magma_int_t k2, const magma_int_t *ipiv, magma_int_t inci, magma_queue_t queue );
```
"""
function magmablas_hlaswp(n, dAT, ldda, k1, k2, ipiv, inci, queue)
    ccall((:magmablas_hlaswp, libmagma), Cvoid, (magma_int_t, Ptr{magmaHalf}, magma_int_t, magma_int_t, magma_int_t, Ptr{magma_int_t}, magma_int_t, magma_queue_t), n, dAT, ldda, k1, k2, ipiv, inci, queue)
end

# no prototype is found for this function at magmablas_v1.h:52:6, please use with caution
"""
    magma_device_sync()

device_sync is not portable to OpenCL, and is generally not needed
### Prototype
```c
void magma_device_sync();
```
"""
function magma_device_sync()
    ccall((:magma_device_sync, libmagma), Cvoid, ())
end

"""
    magmablasSetKernelStream(queue)

=============================================================================
Define magma queue
@deprecated
### Prototype
```c
magma_int_t magmablasSetKernelStream( magma_queue_t queue );
```
"""
function magmablasSetKernelStream(queue)
    ccall((:magmablasSetKernelStream, libmagma), magma_int_t, (magma_queue_t,), queue)
end

"""
    magmablasGetKernelStream(queue)


### Prototype
```c
magma_int_t magmablasGetKernelStream( magma_queue_t *queue );
```
"""
function magmablasGetKernelStream(queue)
    ccall((:magmablasGetKernelStream, libmagma), magma_int_t, (Ptr{magma_queue_t},), queue)
end

# no prototype is found for this function at magmablas_v1.h:60:15, please use with caution
"""
    magmablasGetQueue()


### Prototype
```c
magma_queue_t magmablasGetQueue();
```
"""
function magmablasGetQueue()
    ccall((:magmablasGetQueue, libmagma), magma_queue_t, ())
end

const MAGMA_API = 2

const magmaCfma = cuCfma

const magmaCfmaf = cuCfmaf

const MAGMA_Z_ZERO = MAGMA_Z_MAKE(0.0, 0.0)

const MAGMA_Z_ONE = MAGMA_Z_MAKE(1.0, 0.0)

const MAGMA_Z_HALF = MAGMA_Z_MAKE(0.5, 0.0)

const MAGMA_Z_NEG_ONE = MAGMA_Z_MAKE(-1.0, 0.0)

const MAGMA_Z_NEG_HALF = MAGMA_Z_MAKE(-0.5, 0.0)

const MAGMA_C_ZERO = MAGMA_C_MAKE(0.0, 0.0)

const MAGMA_C_ONE = MAGMA_C_MAKE(1.0, 0.0)

const MAGMA_C_HALF = MAGMA_C_MAKE(0.5, 0.0)

const MAGMA_C_NEG_ONE = MAGMA_C_MAKE(-1.0, 0.0)

const MAGMA_C_NEG_HALF = MAGMA_C_MAKE(-0.5, 0.0)

const MAGMA_D_ZERO = 0.0

const MAGMA_D_ONE = 1.0

const MAGMA_D_HALF = 0.5

const MAGMA_D_NEG_ONE = -1.0

const MAGMA_D_NEG_HALF = -0.5

const MAGMA_S_ZERO = 0.0

const MAGMA_S_ONE = 1.0

const MAGMA_S_HALF = 0.5

const MAGMA_S_NEG_ONE = -1.0

const MAGMA_S_NEG_HALF = -0.5

const MAGMA_VERSION_MAJOR = 2

const MAGMA_VERSION_MINOR = 7

const MAGMA_VERSION_MICRO = 0

const MAGMA_VERSION_STAGE = "svn"

const MagmaMaxGPUs = 8

const MagmaMaxAccelerators = 8

const MagmaMaxSubs = 16

const MagmaBigTileSize = 1000000

const MAGMA_SUCCESS = 0

const MAGMA_ERR = -100

const MAGMA_ERR_NOT_INITIALIZED = -101

const MAGMA_ERR_REINITIALIZED = -102

const MAGMA_ERR_NOT_SUPPORTED = -103

const MAGMA_ERR_ILLEGAL_VALUE = -104

const MAGMA_ERR_NOT_FOUND = -105

const MAGMA_ERR_ALLOCATION = -106

const MAGMA_ERR_INTERNAL_LIMIT = -107

const MAGMA_ERR_UNALLOCATED = -108

const MAGMA_ERR_FILESYSTEM = -109

const MAGMA_ERR_UNEXPECTED = -110

const MAGMA_ERR_SEQUENCE_FLUSHED = -111

const MAGMA_ERR_HOST_ALLOC = -112

const MAGMA_ERR_DEVICE_ALLOC = -113

const MAGMA_ERR_CUDASTREAM = -114

const MAGMA_ERR_INVALID_PTR = -115

const MAGMA_ERR_UNKNOWN = -116

const MAGMA_ERR_NOT_IMPLEMENTED = -117

const MAGMA_ERR_NAN = -118

const MAGMA_SLOW_CONVERGENCE = -201

const MAGMA_DIVERGENCE = -202

const MAGMA_NONSPD = -203

const MAGMA_ERR_BADPRECOND = -204

const MAGMA_NOTCONVERGED = -205

const MAGMA_ERR_CUSPARSE = -3000

const MAGMA_ERR_CUSPARSE_NOT_INITIALIZED = -3001

const MAGMA_ERR_CUSPARSE_ALLOC_FAILED = -3002

const MAGMA_ERR_CUSPARSE_INVALID_VALUE = -3003

const MAGMA_ERR_CUSPARSE_ARCH_MISMATCH = -3004

const MAGMA_ERR_CUSPARSE_MAPPING_ERROR = -3005

const MAGMA_ERR_CUSPARSE_EXECUTION_FAILED = -3006

const MAGMA_ERR_CUSPARSE_INTERNAL_ERROR = -3007

const MAGMA_ERR_CUSPARSE_MATRIX_TYPE_NOT_SUPPORTED = -3008

const MAGMA_ERR_CUSPARSE_ZERO_PIVOT = -3009

const Magma2lapack_Min = MagmaFalse

const Magma2lapack_Max = MagmaRowwise

const MagmaRowMajorStr = "Row"

const MagmaColMajorStr = "Col"

const MagmaNoTransStr = "NoTrans"

const MagmaTransStr = "Trans"

const MagmaConjTransStr = "ConjTrans"

const Magma_ConjTransStr = "ConjTrans"

const MagmaUpperStr = "Upper"

const MagmaLowerStr = "Lower"

const MagmaFullStr = "Full"

const MagmaNonUnitStr = "NonUnit"

const MagmaUnitStr = "Unit"

const MagmaLeftStr = "Left"

const MagmaRightStr = "Right"

const MagmaBothSidesStr = "Both"

const MagmaOneNormStr = "1"

const MagmaTwoNormStr = "2"

const MagmaFrobeniusNormStr = "Fro"

const MagmaInfNormStr = "Inf"

const MagmaMaxNormStr = "Max"

const MagmaForwardStr = "Forward"

const MagmaBackwardStr = "Backward"

const MagmaColumnwiseStr = "Columnwise"

const MagmaRowwiseStr = "Rowwise"

const MagmaNoVecStr = "NoVec"

const MagmaVecStr = "Vec"

const MagmaIVecStr = "IVec"

const MagmaAllVecStr = "All"

const MagmaSomeVecStr = "Some"

const MagmaOverwriteVecStr = "Overwrite"

# Skipping MacroDefinition: magma_setvector ( n , elemSize , hx_src , incx , dy_dst , incy , queue ) magma_setvector_internal ( n , elemSize , hx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_getvector ( n , elemSize , dx_src , incx , hy_dst , incy , queue ) magma_getvector_internal ( n , elemSize , dx_src , incx , hy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_copyvector ( n , elemSize , dx_src , incx , dy_dst , incy , queue ) magma_copyvector_internal ( n , elemSize , dx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_setmatrix ( m , n , elemSize , hA_src , lda , dB_dst , lddb , queue ) magma_setmatrix_internal ( m , n , elemSize , hA_src , lda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_getmatrix ( m , n , elemSize , dA_src , ldda , hB_dst , ldb , queue ) magma_getmatrix_internal ( m , n , elemSize , dA_src , ldda , hB_dst , ldb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_copymatrix ( m , n , elemSize , dA_src , ldda , dB_dst , lddb , queue ) magma_copymatrix_internal ( m , n , elemSize , dA_src , ldda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_isetvector ( n , hx_src , incx , dy_dst , incy , queue ) magma_isetvector_internal ( n , hx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_igetvector ( n , dx_src , incx , hy_dst , incy , queue ) magma_igetvector_internal ( n , dx_src , incx , hy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_icopyvector ( n , dx_src , incx , dy_dst , incy , queue ) magma_icopyvector_internal ( n , dx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_isetmatrix ( m , n , hA_src , lda , dB_dst , lddb , queue ) magma_isetmatrix_internal ( m , n , hA_src , lda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_igetmatrix ( m , n , dA_src , ldda , hB_dst , ldb , queue ) magma_igetmatrix_internal ( m , n , dA_src , ldda , hB_dst , ldb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_icopymatrix ( m , n , dA_src , ldda , dB_dst , lddb , queue ) magma_icopymatrix_internal ( m , n , dA_src , ldda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_index_setvector ( n , hx_src , incx , dy_dst , incy , queue ) magma_index_setvector_internal ( n , hx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_index_getvector ( n , dx_src , incx , hy_dst , incy , queue ) magma_index_getvector_internal ( n , dx_src , incx , hy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_index_copyvector ( n , dx_src , incx , dy_dst , incy , queue ) magma_index_copyvector_internal ( n , dx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_index_setmatrix ( m , n , hA_src , lda , dB_dst , lddb , queue ) magma_index_setmatrix_internal ( m , n , hA_src , lda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_index_getmatrix ( m , n , dA_src , ldda , hB_dst , ldb , queue ) magma_index_getmatrix_internal ( m , n , dA_src , ldda , hB_dst , ldb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_index_copymatrix ( m , n , dA_src , ldda , dB_dst , lddb , queue ) magma_index_copymatrix_internal ( m , n , dA_src , ldda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_zsetvector ( n , hx_src , incx , dy_dst , incy , queue ) magma_zsetvector_internal ( n , hx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_zgetvector ( n , dx_src , incx , hy_dst , incy , queue ) magma_zgetvector_internal ( n , dx_src , incx , hy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_zcopyvector ( n , dx_src , incx , dy_dst , incy , queue ) magma_zcopyvector_internal ( n , dx_src , incx , dy_dst , incy , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_zsetmatrix ( m , n , hA_src , lda , dB_dst , lddb , queue ) magma_zsetmatrix_internal ( m , n , hA_src , lda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_zgetmatrix ( m , n , dA_src , ldda , hB_dst , ldb , queue ) magma_zgetmatrix_internal ( m , n , dA_src , ldda , hB_dst , ldb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_zcopymatrix ( m , n , dA_src , ldda , dB_dst , lddb , queue ) magma_zcopymatrix_internal ( m , n , dA_src , ldda , dB_dst , lddb , queue , __func__ , __FILE__ , __LINE__ )

# Skipping MacroDefinition: magma_queue_create ( device , queue_ptr ) magma_queue_create_internal ( device , queue_ptr , __func__ , __FILE__ , __LINE__ )

const MagmaUpperLower = MagmaFull

const MagmaUpperLowerStr = MagmaFullStr

const magmablas_ztranspose_inplace = magmablas_ztranspose_inplace_v1

const magmablas_ztranspose_conj_inplace = magmablas_ztranspose_conj_inplace_v1

const magmablas_ztranspose = magmablas_ztranspose_v1

const magmablas_ztranspose_conj = magmablas_ztranspose_conj_v1

const magmablas_zgetmatrix_transpose = magmablas_zgetmatrix_transpose_v1

const magmablas_zsetmatrix_transpose = magmablas_zsetmatrix_transpose_v1

const magmablas_zprbt = magmablas_zprbt_v1

const magmablas_zprbt_mv = magmablas_zprbt_mv_v1

const magmablas_zprbt_mtv = magmablas_zprbt_mtv_v1

const magma_zgetmatrix_1D_col_bcyclic = magma_zgetmatrix_1D_col_bcyclic_v1

const magma_zsetmatrix_1D_col_bcyclic = magma_zsetmatrix_1D_col_bcyclic_v1

const magma_zgetmatrix_1D_row_bcyclic = magma_zgetmatrix_1D_row_bcyclic_v1

const magma_zsetmatrix_1D_row_bcyclic = magma_zsetmatrix_1D_row_bcyclic_v1

const magmablas_zgeadd = magmablas_zgeadd_v1

const magmablas_zgeadd2 = magmablas_zgeadd2_v1

const magmablas_zlacpy = magmablas_zlacpy_v1

const magmablas_zlacpy_conj = magmablas_zlacpy_conj_v1

const magmablas_zlacpy_sym_in = magmablas_zlacpy_sym_in_v1

const magmablas_zlacpy_sym_out = magmablas_zlacpy_sym_out_v1

const magmablas_zlange = magmablas_zlange_v1

const magmablas_zlanhe = magmablas_zlanhe_v1

const magmablas_zlansy = magmablas_zlansy_v1

const magmablas_zlarfg = magmablas_zlarfg_v1

const magmablas_zlascl = magmablas_zlascl_v1

const magmablas_zlascl_2x2 = magmablas_zlascl_2x2_v1

const magmablas_zlascl2 = magmablas_zlascl2_v1

const magmablas_zlascl_diag = magmablas_zlascl_diag_v1

const magmablas_zlaset = magmablas_zlaset_v1

const magmablas_zlaset_band = magmablas_zlaset_band_v1

const magmablas_zlaswp = magmablas_zlaswp_v1

const magmablas_zlaswp2 = magmablas_zlaswp2_v1

const magmablas_zlaswp_sym = magmablas_zlaswp_sym_v1

const magmablas_zlaswpx = magmablas_zlaswpx_v1

const magmablas_zsymmetrize = magmablas_zsymmetrize_v1

const magmablas_zsymmetrize_tiles = magmablas_zsymmetrize_tiles_v1

const magmablas_ztrtri_diag = magmablas_ztrtri_diag_v1

const magmablas_dznrm2_adjust = magmablas_dznrm2_adjust_v1

const magmablas_dznrm2_check = magmablas_dznrm2_check_v1

const magmablas_dznrm2_cols = magmablas_dznrm2_cols_v1

const magmablas_dznrm2_row_check_adjust = magmablas_dznrm2_row_check_adjust_v1

const magma_zlarfb_gpu = magma_zlarfb_gpu_v1

const magma_zlarfb_gpu_gemm = magma_zlarfb_gpu_gemm_v1

const magma_zlarfbx_gpu = magma_zlarfbx_gpu_v1

const magma_zlarfg_gpu = magma_zlarfg_gpu_v1

const magma_zlarfgtx_gpu = magma_zlarfgtx_gpu_v1

const magma_zlarfgx_gpu = magma_zlarfgx_gpu_v1

const magma_zlarfx_gpu = magma_zlarfx_gpu_v1

const magmablas_zaxpycp = magmablas_zaxpycp_v1

const magmablas_zswap = magmablas_zswap_v1

const magmablas_zswapblk = magmablas_zswapblk_v1

const magmablas_zswapdblk = magmablas_zswapdblk_v1

const magmablas_zgemv = magmablas_zgemv_v1

const magmablas_zgemv_conj = magmablas_zgemv_conj_v1

const magmablas_zhemv = magmablas_zhemv_v1

const magmablas_zsymv = magmablas_zsymv_v1

const magmablas_zgemm = magmablas_zgemm_v1

const magmablas_zgemm_reduce = magmablas_zgemm_reduce_v1

const magmablas_zhemm = magmablas_zhemm_v1

const magmablas_zsymm = magmablas_zsymm_v1

const magmablas_zsyr2k = magmablas_zsyr2k_v1

const magmablas_zher2k = magmablas_zher2k_v1

const magmablas_zsyrk = magmablas_zsyrk_v1

const magmablas_zherk = magmablas_zherk_v1

const magmablas_ztrsm = magmablas_ztrsm_v1

const magmablas_ztrsm_outofplace = magmablas_ztrsm_outofplace_v1

const magmablas_ztrsm_work = magmablas_ztrsm_work_v1

const magma_izamax = magma_izamax_v1

const magma_izamin = magma_izamin_v1

const magma_dzasum = magma_dzasum_v1

const magma_zaxpy = magma_zaxpy_v1

const magma_zcopy = magma_zcopy_v1

const magma_zdotc = magma_zdotc_v1

const magma_zdotu = magma_zdotu_v1

const magma_dznrm2 = magma_dznrm2_v1

const magma_zrot = magma_zrot_v1

const magma_zdrot = magma_zdrot_v1

const magma_zrotm = magma_zrotm_v1

const magma_zrotmg = magma_zrotmg_v1

const magma_zscal = magma_zscal_v1

const magma_zdscal = magma_zdscal_v1

const magma_zswap = magma_zswap_v1

const magma_zgemv = magma_zgemv_v1

const magma_zgerc = magma_zgerc_v1

const magma_zgeru = magma_zgeru_v1

const magma_zhemv = magma_zhemv_v1

const magma_zher = magma_zher_v1

const magma_zher2 = magma_zher2_v1

const magma_ztrmv = magma_ztrmv_v1

const magma_ztrsv = magma_ztrsv_v1

const magma_zgemm = magma_zgemm_v1

const magma_zsymm = magma_zsymm_v1

const magma_zhemm = magma_zhemm_v1

const magma_zsyr2k = magma_zsyr2k_v1

const magma_zher2k = magma_zher2k_v1

const magma_zsyrk = magma_zsyrk_v1

const magma_zherk = magma_zherk_v1

const magma_ztrmm = magma_ztrmm_v1

const magma_ztrsm = magma_ztrsm_v1

const magmablas_zcaxpycp = magmablas_zcaxpycp_v1

const magmablas_zclaswp = magmablas_zclaswp_v1

const magmablas_zlag2c = magmablas_zlag2c_v1

const magmablas_clag2z = magmablas_clag2z_v1

const magmablas_zlat2c = magmablas_zlat2c_v1

const magmablas_clat2z = magmablas_clat2z_v1

end # module
