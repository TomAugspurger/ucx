/*
 *  Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *  See COPYRIGHT for license information
 */

#include "cudaipc_copy_ep.h"
#include "cudaipc_copy_iface.h"
#include "cudaipc_copy_md.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>

SGLIB_DEFINE_LIST_FUNCTIONS(uct_cudaipc_copy_rem_seg_t,
                            uct_cudaipc_copy_rem_seg_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_cudaipc_copy_rem_seg_t,
                                        UCT_CUDAIPC_COPY_HASH_SIZE,
                                        uct_cudaipc_copy_rem_seg_hash)

static UCS_CLASS_INIT_FUNC(uct_cudaipc_copy_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_cudaipc_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                     uct_cudaipc_copy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    sglib_hashed_uct_cudaipc_copy_rem_seg_t_init(self->rem_segments_hash);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cudaipc_copy_ep_t)
{
    uct_cudaipc_copy_rem_seg_t *remote_seg;
    CUresult                   cu_ret;
    struct sglib_hashed_uct_cudaipc_copy_rem_seg_t_iterator iter;

    for (remote_seg = sglib_hashed_uct_cudaipc_copy_rem_seg_t_it_init(&iter, self->rem_segments_hash);
         remote_seg != NULL;
         remote_seg = sglib_hashed_uct_cudaipc_copy_rem_seg_t_it_next(&iter)) {
            sglib_hashed_uct_cudaipc_copy_rem_seg_t_delete(self->rem_segments_hash,
                                                           remote_seg);
            cu_ret = cuIpcCloseMemHandle(remote_seg->d_bptr);
            if (cu_ret != CUDA_SUCCESS) {
                cuGetErrorString(cu_ret, &cu_err_str);
                ucs_fatal("cuIpcCloseMemHandle failed. d_ptr :%p ret:%s",
                          (void *) remote_seg->d_bptr, cu_err_str);
            }
            ucs_free(remote_seg);
    }
}

UCS_CLASS_DEFINE(uct_cudaipc_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cudaipc_copy_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cudaipc_copy_ep_t, uct_ep_t);

#define uct_cudaipc_copy_trace_data(_remote_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                   (_rkey))

#define UCT_CUDAIPC_ZERO_LENGTH_POST(len)                       \
    do {                                                        \
        if (0 == len) {                                         \
            ucs_trace_data("Zero length request: skip it");     \
            return UCS_OK;                                      \
        }                                                       \
    } while(0);

void *uct_cudaipc_copy_ep_attach_rem_seg(uct_cudaipc_copy_ep_t *ep,
                                         uct_cudaipc_copy_iface_t *iface,
                                         uct_cudaipc_copy_key_t *rkey)
{
    uct_cudaipc_copy_rem_seg_t *rem_seg, search;
    CUdevice                   cu_device;
    CUresult                   cu_ret = CUDA_SUCCESS;

    cu_ret = cuCtxGetDevice(&cu_device);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_fatal("cuCtxGetDevice failed ret:%s", cu_err_str);
    }

    /* Are all other members of *search* zeroed out? or ignored?*/
    search.ph = rkey->ph;
    rem_seg = sglib_hashed_uct_cudaipc_copy_rem_seg_t_find_member(ep->rem_segments_hash, &search);
    if (rem_seg == NULL) {
        rem_seg = ucs_malloc(sizeof(*rem_seg), "rem_seg");
        if (rem_seg == NULL) {
            ucs_fatal("Failed to allocated memory for a remote segment. %m");
        }

        rem_seg->ph      = rkey->ph;
        rem_seg->dev_num = rkey->dev_num;
        /* Attach memory to own GPU address space */
        cu_ret = cuIpcOpenMemHandle((CUdeviceptr *) &rem_seg->d_bptr,
                                    rkey->ph,
                                    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
        if (CUDA_SUCCESS != cu_ret) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_fatal("cuIpcOpenMemHandle failed: ret:%s", cu_err_str);
        }
        rem_seg->b_len = rkey->b_rem_len;

        /* put the base address into the ep's hash table */
        sglib_hashed_uct_cudaipc_copy_rem_seg_t_add(ep->rem_segments_hash,
                                                    rem_seg);
    }

    return (void *) rem_seg->d_bptr;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cudaipc_copy_post_cuda_async_copy(uct_ep_h tl_ep, void *dst, void *src,
                                      size_t length, CUstream stream,
                                      ucs_queue_head_t *outstanding_queue,
                                      uct_completion_t *comp)
{
    uct_cudaipc_copy_iface_t      *iface;
    uct_cudaipc_copy_event_desc_t *cudaipc_event;
    CUresult                       cu_ret = CUDA_SUCCESS;

    iface = ucs_derived_of(tl_ep->iface, uct_cudaipc_copy_iface_t);

    if (!length) {
        return UCS_OK;
    }

    cudaipc_event = ucs_mpool_get(&iface->cudaipc_event_desc);
    if (ucs_unlikely(cudaipc_event == NULL)) {
        ucs_error("Failed to allocate cudaipc event object");
        return UCS_ERR_NO_MEMORY;
    }

    cu_ret = cuMemcpyDtoDAsync((CUdeviceptr) dst, (CUdeviceptr) src, length,
                               stream);
    if (CUDA_SUCCESS != cu_ret) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuMemcpyDtoD Failed ret:%s", cu_err_str);
        return UCS_ERR_IO_ERROR;
    }

    cu_ret = cuEventRecord(cudaipc_event->event, stream);
    if (CUDA_SUCCESS != cu_ret) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuEventRecord Failed ret:%s", cu_err_str);
        return UCS_ERR_IO_ERROR;
    }
    ucs_queue_push(outstanding_queue, &cudaipc_event->queue);
    cudaipc_event->comp = comp;

    ucs_info("cuMemcpyDtoDAsync issued :%p dst:%p, src:%p  len:%ld",
             cudaipc_event, dst, src, length);
    return UCS_INPROGRESS;
}

ucs_status_t uct_cudaipc_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                           size_t iovcnt,
                                           uint64_t remote_addr,
                                           uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    uct_cudaipc_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                     uct_cudaipc_copy_iface_t);
    uct_cudaipc_copy_key_t   *key   = (uct_cudaipc_copy_key_t *) rkey;
    ucs_status_t             status = UCS_OK;
    CUdevice                 cu_device;
    CUresult                 cu_ret = CUDA_SUCCESS;
    CUresult                 cu_err = CUDA_SUCCESS;
    int                      offset = 0;
    void                     *mapped_rem_addr      = NULL;
    void                     *mapped_rem_base_addr = NULL;
    CUcontext                local_ptr_ctx;
    CUcontext                remote_ptr_ctx;
    CUpointer_attribute      attribute;
    uct_cudaipc_copy_ep_t    *ep = ucs_derived_of(tl_ep, uct_cudaipc_copy_ep_t);

    UCT_CUDAIPC_ZERO_LENGTH_POST(iov[0].length);

    cu_ret = cuCtxGetDevice(&cu_device);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuCtxGetDevice failed ret:%s", cu_err_str);
        goto err;
    }

    if (key->dev_num == (int) cu_device) {

        attribute = CU_POINTER_ATTRIBUTE_CONTEXT;
        cu_ret = cuPointerGetAttribute((void *) &remote_ptr_ctx, attribute,
                                       (CUdeviceptr) remote_addr);
        if (CUDA_ERROR_INVALID_VALUE != cu_ret) {
            /* context belongs to another process that uses the same
               device */
            cu_err = cu_ret;
        }
        else if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuPointerGetAttribute failed ret:%s", cu_err_str);
            goto err;
        }

        /* assumes iov is uniformly on device or not */
        cu_ret = cuPointerGetAttribute((void *) &local_ptr_ctx, attribute,
                                       (CUdeviceptr) iov[0].buffer);
        if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuPointerGetAttribute failed ret:%s", cu_err_str);
            goto err;
        }
    }

    if (0 == iface->streams_initialized) {
        status = uct_cudaipc_copy_iface_init_streams(iface);
        if (UCS_OK != status) return status;
    }

    if ((CUDA_SUCCESS == cu_err) && (key->dev_num == (int) cu_device) &&
        (local_ptr_ctx == remote_ptr_ctx)) {
        mapped_rem_addr = (void *) remote_addr;
    }
    else {
        /* Is uintptr_t equivalent to uint64_t?  */
        mapped_rem_base_addr = uct_cudaipc_copy_ep_attach_rem_seg(ep, iface,
                                                                  key);
        offset = (uintptr_t) remote_addr - (uintptr_t) key->d_rem_bptr;
        if (offset > key->b_rem_len) {
            ucs_fatal("Attempting to access memory outside memory range\n");
        }
        mapped_rem_addr = (void *) ((uintptr_t) mapped_rem_base_addr + offset);
    }

    status = uct_cudaipc_copy_post_cuda_async_copy(tl_ep, iov[0].buffer,
                                                   (void *) mapped_rem_addr,
                                                   iov[0].length,
                                                   iface->stream_d2d[key->dev_num],
                                                   &iface->outstanding_d2d_cudaipc_event_q, comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cudaipc_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                                uct_iov_total_length(iov, iovcnt));
    return status;
 err:
    return UCS_ERR_IO_ERROR;
}

ucs_status_t uct_cudaipc_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                           size_t iovcnt,
                                           uint64_t remote_addr,
                                           uct_rkey_t rkey,
                                           uct_completion_t *comp)
{

    uct_cudaipc_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                     uct_cudaipc_copy_iface_t);
    uct_cudaipc_copy_key_t   *key   = (uct_cudaipc_copy_key_t *) rkey;
    ucs_status_t             status = UCS_OK;
    CUdevice                 cu_device;
    CUresult                 cu_ret = CUDA_SUCCESS;
    CUresult                 cu_err = CUDA_SUCCESS;
    int                      offset = 0;
    void                     *mapped_rem_addr      = NULL;
    void                     *mapped_rem_base_addr = NULL;
    CUcontext                local_ptr_ctx;
    CUcontext                remote_ptr_ctx;
    CUpointer_attribute      attribute;
    uct_cudaipc_copy_ep_t    *ep = ucs_derived_of(tl_ep, uct_cudaipc_copy_ep_t);

    UCT_CUDAIPC_ZERO_LENGTH_POST(iov[0].length);

    cu_ret = cuCtxGetDevice(&cu_device);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuCtxGetDevice failed ret:%s", cu_err_str);
        goto err;
    }

    if (key->dev_num == (int) cu_device) {

        attribute = CU_POINTER_ATTRIBUTE_CONTEXT;
        cu_ret = cuPointerGetAttribute((void *) &remote_ptr_ctx, attribute,
                                       (CUdeviceptr) remote_addr);
        if (CUDA_ERROR_INVALID_VALUE != cu_ret) {
            /* context belongs to another process that uses the same
               device */
            cu_err = cu_ret;
        }
        else if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuPointerGetAttribute failed ret:%s", cu_err_str);
            goto err;
        }

        /* assumes iov is uniformly on device or not */
        cu_ret = cuPointerGetAttribute((void *) &local_ptr_ctx, attribute,
                                       (CUdeviceptr) iov[0].buffer);
        if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuPointerGetAttribute failed ret:%s", cu_err_str);
            goto err;
        }
    }

    if (0 == iface->streams_initialized) {
        status = uct_cudaipc_copy_iface_init_streams(iface);
        if (UCS_OK != status) return status;
    }

    if ((CUDA_SUCCESS == cu_err) && (key->dev_num == (int) cu_device) &&
        (local_ptr_ctx == remote_ptr_ctx)) {
        mapped_rem_addr = (void *) remote_addr;
    }
    else {
        /* Is uintptr_t equivalent to uint64_t?  */
        mapped_rem_base_addr = uct_cudaipc_copy_ep_attach_rem_seg(ep, iface,
                                                                  key);
        offset = (uintptr_t) remote_addr - (uintptr_t) key->d_rem_bptr;
        if (offset > key->b_rem_len) {
            ucs_fatal("Attempting to access memory outside memory range\n");
        }
        mapped_rem_addr = (void *) ((uintptr_t) mapped_rem_base_addr + offset);
    }

    status = uct_cudaipc_copy_post_cuda_async_copy(tl_ep,
                                                   (void *) mapped_rem_addr,
                                                   iov[0].buffer,
                                                   iov[0].length,
                                                   iface->stream_d2d[key->dev_num],
                                                   &iface->outstanding_d2d_cudaipc_event_q,
                                                   comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cudaipc_copy_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                                uct_iov_total_length(iov, iovcnt));
    return status;
 err:
    return UCS_ERR_IO_ERROR;
}
