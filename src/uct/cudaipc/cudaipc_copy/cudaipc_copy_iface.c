/*
 *  Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *  See COPYRIGHT for license information
 */

#include "cudaipc_copy_iface.h"
#include "cudaipc_copy_md.h"
#include "cudaipc_copy_ep.h"

#include <ucs/type/class.h>
#include <ucs/sys/string.h>

static ucs_config_field_t uct_cudaipc_copy_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_cudaipc_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16",
     "Max number of event completions to pick during cuda events polling",
      ucs_offsetof(uct_cudaipc_copy_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cudaipc_copy_iface_t)(uct_iface_t*);

static ucs_status_t uct_cudaipc_copy_iface_get_address(uct_iface_h tl_iface,
                                                       uct_iface_addr_t *iface_addr)
{
    uint64_t *cudaipc_copy_addr = (uint64_t *)iface_addr;

    *cudaipc_copy_addr = ucs_machine_guid();
    return UCS_OK;
}

/*
 * Check if another iface is reachable *in principle*
 *
 * Limiting reachability to same socket doesn't work because on
 * systems with PCI switches, GPUs may be alltoall peer-reachable.
 *
 * Returning 1 if processes belong to same machine for
 * now because from the time this function is called to the time when
 * a transfer operation is requested, the pair of devices being used
 * by the two processes may be different.
 *
 * Corner case: if there are over 8 GPUs that are peer accessible,
 * some of them may not be reachable
 *
 * Instead processes use rkey to check if peer is accessible at that
 * point and transfer attempt may fail
 */
static int uct_cudaipc_copy_iface_is_reachable(const uct_iface_h iface,
                                               const uct_device_addr_t *dev_addr,
                                               const uct_iface_addr_t *iface_addr)
{
    return (ucs_machine_guid() == *((uint64_t *)iface_addr));
}

static ucs_status_t uct_cudaipc_copy_iface_query(uct_iface_h iface,
                                                 uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    iface_attr->iface_addr_len          = sizeof(uint64_t);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_PENDING   |
                                          UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY;

    iface_attr->cap.put.max_short       = 0;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = UCT_CUDAIPC_MAX_ALLOC_SZ; /*SIZE_MAX;*/
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = UCT_CUDAIPC_MAX_ALLOC_SZ; /*SIZE_MAX;*/
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency.overhead        = 1e-9; /* FIXME */
    iface_attr->latency.growth          = 0; /* FIXME */
    iface_attr->bandwidth               = 6911 * 1024.0 * 1024.0; /* FIXME */
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t uct_cudaipc_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                                 uct_completion_t *comp)
{
    uct_cudaipc_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cudaipc_copy_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (ucs_queue_is_empty(&iface->outstanding_d2d_cudaipc_event_q) &&
        ucs_queue_is_empty(&iface->outstanding_h2d_cudaipc_event_q) &&
        ucs_queue_is_empty(&iface->outstanding_d2h_cudaipc_event_q)) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_cudaipc_copy_progress_event_queue(ucs_queue_head_t *event_queue, unsigned max_events)
{
    unsigned                       count = 0;
    CUresult                       cu_ret = CUDA_SUCCESS;
    uct_cudaipc_copy_event_desc_t *cudaipc_event;
    ucs_queue_iter_t               iter;

    ucs_queue_for_each_safe(cudaipc_event, iter, event_queue, queue) {
        cu_ret = cuEventQuery(cudaipc_event->event);
        if (CUDA_SUCCESS != cu_ret) {
            break;
        }
        ucs_queue_del_iter(event_queue, iter);
        if (cudaipc_event->comp != NULL) {
            uct_invoke_completion(cudaipc_event->comp, UCS_OK);
        }
        ucs_trace_poll("CUDAIPC Event Done :%p", cudaipc_event);
        ucs_mpool_put(cudaipc_event);
        count++;
        if (count >= max_events) {
            break;
        }
    }
    return count;
}

static unsigned uct_cudaipc_copy_iface_progress(uct_iface_h tl_iface)
{
    uct_cudaipc_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cudaipc_copy_iface_t);
    unsigned                  max_events = iface->config.max_poll;
    unsigned                  count;

    count = uct_cudaipc_copy_progress_event_queue(&iface->outstanding_d2d_cudaipc_event_q,
                                                  max_events);
    count += uct_cudaipc_copy_progress_event_queue(&iface->outstanding_d2h_cudaipc_event_q,
                                                   max_events);
    count += uct_cudaipc_copy_progress_event_queue(&iface->outstanding_h2d_cudaipc_event_q,
                                                   max_events);
    return count;
}

static uct_iface_ops_t uct_cudaipc_copy_iface_ops = {
    .ep_get_zcopy             = uct_cudaipc_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_cudaipc_copy_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_cudaipc_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cudaipc_copy_ep_t),
    .iface_flush              = uct_cudaipc_copy_iface_flush,
    .iface_fence              = uct_base_iface_fence, /*TODO*/
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_cudaipc_copy_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cudaipc_copy_iface_t),
    .iface_query              = uct_cudaipc_copy_iface_query,
    .iface_get_device_address = (void*)ucs_empty_function_return_success,
    .iface_get_address        = uct_cudaipc_copy_iface_get_address,
    .iface_is_reachable       = uct_cudaipc_copy_iface_is_reachable,
};

static void uct_cudaipc_copy_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cudaipc_copy_event_desc_t *base = (uct_cudaipc_copy_event_desc_t *) obj;
    CUresult                       cu_ret;

    memset(base, 0 , sizeof(*base));
    cu_ret = cuEventCreate(&(base->event), CU_EVENT_DISABLE_TIMING);
    if (CUDA_SUCCESS != cu_ret) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuEventCreate Failed ret:%s", cu_err_str);
    }
}

static void uct_cudaipc_copy_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_cudaipc_copy_event_desc_t *base = (uct_cudaipc_copy_event_desc_t *) obj;
    CUresult                       cu_ret;

    cu_ret = cuEventDestroy(base->event);
    if (CUDA_SUCCESS != cu_ret) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuEventDestroy Failed ret:%s", cu_err_str);
    }
}

ucs_status_t uct_cudaipc_copy_iface_init_streams(uct_cudaipc_copy_iface_t *iface)
{
    int       i;
    CUresult  cu_ret;

    for (i = 0; i < iface->device_count; i++) {
        cu_ret = cuStreamCreate(&iface->stream_d2d[i], CU_STREAM_NON_BLOCKING);
        if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuStreamCreate d2d (lane %d) error; ret:%s", i,
                      cu_err_str);
            return UCS_ERR_IO_ERROR;
        }
        cu_ret = cuStreamCreate(&iface->stream_d2h[i], CU_STREAM_NON_BLOCKING);

        if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuStreamCreate d2h (lane %d) error; ret:%s", i,
                      cu_err_str);
            return UCS_ERR_IO_ERROR;
        }
        cu_ret = cuStreamCreate(&iface->stream_h2d[i], CU_STREAM_NON_BLOCKING);

        if (cu_ret != CUDA_SUCCESS) {
            cuGetErrorString(cu_ret, &cu_err_str);
            ucs_error("cuStreamCreate h2d (lane %d) error; ret:%s", i,
                      cu_err_str);
            return UCS_ERR_IO_ERROR;
        }
    }
    iface->streams_initialized = 1;

    return UCS_OK;
}

static ucs_mpool_ops_t uct_cudaipc_copy_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_cudaipc_copy_event_desc_init,
    .obj_cleanup   = uct_cudaipc_copy_event_desc_cleanup,
};

static UCS_CLASS_INIT_FUNC(uct_cudaipc_copy_iface_t, uct_md_h md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cudaipc_copy_iface_config_t *config = ucs_derived_of(tl_config,
                                                             uct_cudaipc_copy_iface_config_t);
    ucs_status_t                     status;
    CUresult                         cu_ret;
    int                              dev_count;
    int                              i = 0, j = 0;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cudaipc_copy_iface_ops, md, worker,
                              params, tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_CUDAIPC_COPY_TL_NAME));

    if (strncmp(params->mode.device.dev_name,
                UCT_CUDAIPC_DEV_NAME, strlen(UCT_CUDAIPC_DEV_NAME)) != 0) {
        ucs_error("No device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    for (i = 0; i < UCT_CUDAIPC_MAX_PEERS; i++) {
        for (i = 0; i < UCT_CUDAIPC_MAX_PEERS; i++) {
            self->cudaipc_p2p_map[i][j] = -1;
        }
    }

    cu_ret = cuDeviceGetCount(&dev_count);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuDeviceGetCount error - ret:%s", cu_err_str);
        goto err;
    }
    self->device_count = dev_count;

    for (i = 0; i < dev_count; i++) {
        for (j = 0; j < dev_count; j++) {
            cu_ret = cuDeviceCanAccessPeer(&(self->cudaipc_p2p_map[i][j]),
                                           (CUdevice) i, (CUdevice) j);

            if (cu_ret != CUDA_SUCCESS) {
                cuGetErrorString(cu_ret, &cu_err_str);
                ucs_error("cuDeviceCanAccessPeer error(%d, %d) - ret:%s",
                          i, j, cu_err_str);
                goto err;
            }
        }
    }

    ucs_trace("cudaipc gpu p2p map generated for %d devices", dev_count);

    self->config.max_poll = config->max_poll;

    status = ucs_mpool_init(&self->cudaipc_event_desc,
                            0,
                            sizeof(uct_cudaipc_copy_event_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            1024,
                            &uct_cudaipc_copy_event_desc_mpool_ops,
                            "CUDAIPC EVENT objects");

    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }

    self->streams_initialized = 0;

    ucs_queue_head_init(&self->outstanding_d2d_cudaipc_event_q);
    ucs_queue_head_init(&self->outstanding_h2d_cudaipc_event_q);
    ucs_queue_head_init(&self->outstanding_d2h_cudaipc_event_q);

    return UCS_OK;
err:
    return UCS_ERR_IO_ERROR;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cudaipc_copy_iface_t)
{
    CUresult cu_ret;
    int      i;

    if (1 == self->streams_initialized) {
        for (i = 0; i < self->device_count; i++) {
            cu_ret = cuStreamDestroy(self->stream_d2d[i]);

            if (cu_ret != CUDA_SUCCESS) {
                cuGetErrorString(cu_ret, &cu_err_str);
                ucs_error("cuStreamDestroy d2d (lane %d) error; ret:%s", i,
                          cu_err_str);
            }
            cu_ret = cuStreamDestroy(self->stream_d2h[i]);

            if (cu_ret != CUDA_SUCCESS) {
                cuGetErrorString(cu_ret, &cu_err_str);
                ucs_error("cuStreamDestroy d2h (lane %d) error; ret:%s", i,
                          cu_err_str);
            }
            cu_ret = cuStreamDestroy(self->stream_h2d[i]);

            if (cu_ret != CUDA_SUCCESS) {
                cuGetErrorString(cu_ret, &cu_err_str);
                ucs_error("cuStreamDestroy h2d (lane %d) error; ret:%s", i,
                          cu_err_str);
            }
        }
        self->streams_initialized = 0;
    }

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_mpool_cleanup(&self->cudaipc_event_desc, 1);
}

UCS_CLASS_DEFINE(uct_cudaipc_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cudaipc_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cudaipc_copy_iface_t, uct_iface_t);


static ucs_status_t uct_cudaipc_copy_query_tl_resources(uct_md_h md,
                                                        uct_tl_resource_desc_t **resource_p,
                                                        unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resource;

    resource = ucs_calloc(1, sizeof(uct_tl_resource_desc_t), "resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->tl_name, sizeof(resource->tl_name), "%s",
                      UCT_CUDAIPC_COPY_TL_NAME);
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s",
                      UCT_CUDAIPC_DEV_NAME);
    resource->dev_type = UCT_DEVICE_TYPE_ACC;

    *num_resources_p = 1;
    *resource_p      = resource;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_cudaipc_copy_tl,
                        uct_cudaipc_copy_query_tl_resources,
                        uct_cudaipc_copy_iface_t,
                        UCT_CUDAIPC_COPY_TL_NAME,
                        "CUDAIPC_COPY_",
                        uct_cudaipc_copy_iface_config_table,
                        uct_cudaipc_copy_iface_config_t);
UCT_MD_REGISTER_TL(&uct_cudaipc_copy_md_component, &uct_cudaipc_copy_tl);
