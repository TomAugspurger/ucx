# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import cupy as cp
from libc.stdint cimport uintptr_t

cdef extern from "buffer_ops.h":
    int set_device(int)
    data_buf* populate_buffer_region(void *)
    data_buf* populate_buffer_region_with_ptr(unsigned long long int)
    void* return_ptr_from_buf(data_buf*)
    data_buf* allocate_host_buffer(int)
    data_buf* allocate_cuda_buffer(int)
    int free_host_buffer(data_buf*)
    int free_cuda_buffer(data_buf*)
    int set_host_buffer(data_buf*, int, int)
    int set_cuda_buffer(data_buf*, int, int)
    int check_host_buffer(data_buf*, int, int)
    int check_cuda_buffer(data_buf*, int, int)

cdef class buffer_region:
    cdef data_buf* buf
    cdef int is_cuda
    cdef uintptr_t cupy_ptr

    def __cinit__(self):
        return

    def alloc_host(self, len):
        self.buf = allocate_host_buffer(len)
        self.is_cuda = 0

    def alloc_cuda(self, len):
        self.buf = allocate_cuda_buffer(len)
        self.is_cuda = 1

    def free_host(self):
        free_host_buffer(self.buf)

    def free_cuda(self):
        free_cuda_buffer(self.buf)

    def populate_ptr(self, pyobj):
        if type(pyobj) == cp.core.core.ndarray:
            if 'strides' not in pyobj.__cuda_array_interface__:
                self.buf = populate_buffer_region_with_ptr(pyobj.__cuda_array_interface__['data'][0])
            else:
                raise NotImplementedError('non-contiguous arrays are not supported')
        else:
            self.buf = populate_buffer_region(<void *> pyobj)

    def return_obj(self):
        return <object> return_ptr_from_buf(self.buf)
