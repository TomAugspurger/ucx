/*
 *  Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *  See COPYRIGHT for license information
 */

#include "cudaipc_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>

#define UCT_CUDAIPC_COPY_MD_RCACHE_DEFAULT_ALIGN 4096

/*TODO: Need to find actual values for registration overhead and growth*/
static ucs_config_field_t uct_cudaipc_copy_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cudaipc_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"RCACHE", "try", "Enable using memory registration cache",
     ucs_offsetof(uct_cudaipc_copy_md_config_t, enable_rcache), UCS_CONFIG_TYPE_TERNARY},

    {"", "RCACHE_ADDR_ALIGN=" UCS_PP_MAKE_STRING(UCT_CUDAIPC_COPY_MD_RCACHE_DEFAULT_ALIGN), NULL,
     ucs_offsetof(uct_cudaipc_copy_md_config_t, rcache),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_rcache_table)},

    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* Using registration cost for size 0 */
     ucs_offsetof(uct_cudaipc_copy_md_config_t, uc_reg_cost.overhead), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate",
     ucs_offsetof(uct_cudaipc_copy_md_config_t, uc_reg_cost.growth), UCS_CONFIG_TYPE_TIME},

    {NULL}
};

/*TODO: Need to find actual values for registration overhead and growth*/
static ucs_status_t uct_cudaipc_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_CUDA);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_CUDA;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = UCT_CUDAIPC_MAX_ALLOC_SZ; /*ULONG_MAX;*/
    md_attr->rkey_packed_size  = sizeof(uct_cudaipc_copy_key_t);
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

/*TODO: Does remote side need the source ptr?*/
static ucs_status_t uct_cudaipc_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                               void *rkey_buffer)
{
    uct_cudaipc_copy_key_t *packed   = (uct_cudaipc_copy_key_t *) rkey_buffer;
    uct_cudaipc_copy_mem_t *mem_hndl = (uct_cudaipc_copy_mem_t *) memh;

    packed->ph         = mem_hndl->ph;
    packed->d_rem_ptr  = mem_hndl->d_ptr;
    packed->d_rem_bptr = mem_hndl->d_bptr;
    packed->b_rem_len  = mem_hndl->b_len;
    packed->dev_num    = mem_hndl->dev_num;

    return UCS_OK;
}

static ucs_status_t uct_cudaipc_copy_rkey_unpack(uct_md_component_t *mdc,
                                                 const void *rkey_buffer, uct_rkey_t *rkey_p,
                                                 void **handle_p)
{
    uct_cudaipc_copy_key_t *packed = (uct_cudaipc_copy_key_t *) rkey_buffer;
    uct_cudaipc_copy_key_t *key;
    CUdevice               cu_device;
    CUresult               cu_ret = CUDA_SUCCESS;

    cu_ret = cuCtxGetDevice(&cu_device);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuCtxGetDevice failed ret:%s", cu_err_str);
        goto err;
    }

    key = ucs_malloc(sizeof(uct_cudaipc_copy_key_t), "uct_cudaipc_copy_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cudaipc_copy_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    *key = *packed;

    *handle_p = NULL;
    *rkey_p   = (uintptr_t) key;

    return UCS_OK;
 err:
    return UCS_ERR_IO_ERROR;
}

static ucs_status_t uct_cudaipc_copy_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                                  void *handle)
{
    ucs_assert(NULL == handle);

    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t
uct_cudaipc_copy_mem_reg_internal(uct_md_h uct_md, void *address, size_t length,
                                  unsigned flags, uct_cudaipc_copy_mem_t *mem_hndl)
{
    CUresult       cu_ret;
    CUdevice       cu_device;

    if (!length) {
        return UCS_OK;
    }

    cu_ret = cuIpcGetMemHandle(&(mem_hndl->ph), (CUdeviceptr) address);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuIpcGetMemHandle failed. length :%lu ret:%s", length, cu_err_str);
        goto err;
    }

    /* TODO: Following logic doesn't handle the case when multiple
       ctxs are used by the same process */
    cu_ret = cuCtxGetDevice(&cu_device);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuCtxGetDevice failed ret:%s", cu_err_str);
        goto err;
    }

    cu_ret = cuMemGetAddressRange(&(mem_hndl->d_bptr), &(mem_hndl->b_len),
                                  (CUdeviceptr) address);
    if (cu_ret != CUDA_SUCCESS) {
        cuGetErrorString(cu_ret, &cu_err_str);
        ucs_error("cuMemGetAddressRange failed ret:%s", cu_err_str);
        goto err;
    }

    mem_hndl->d_ptr    = (CUdeviceptr) address;
    mem_hndl->reg_size = length;
    mem_hndl->dev_num  = (int) cu_device;

    ucs_trace("registered memory:%p..%p length:%lu d_ptr:%p dev_num:%d",
              address, address + length, length, address, (int) cu_device);

    return UCS_OK;

err:
    return UCS_ERR_IO_ERROR;
}

static ucs_status_t uct_cudaipc_copy_mem_dereg_internal(uct_md_h uct_md, uct_cudaipc_copy_mem_t *mem_hndl)
{
    return UCS_OK;
}

static ucs_status_t uct_cudaipc_copy_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                             unsigned flags, uct_mem_h *memh_p)
{
    uct_cudaipc_copy_mem_t *mem_hndl = NULL;
    ucs_status_t status;

    mem_hndl = ucs_malloc(sizeof(uct_cudaipc_copy_mem_t), "cudaipc_copy handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for cudaipc_copy_mem_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_cudaipc_copy_mem_reg_internal(uct_md, address, length, 0, mem_hndl);
    if (status != UCS_OK) {
        ucs_free(mem_hndl);
        return status;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_cudaipc_copy_mem_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_cudaipc_copy_mem_t *mem_hndl = memh;
    ucs_status_t           status;

    status = uct_cudaipc_copy_mem_dereg_internal(uct_md, mem_hndl);
    if (status != UCS_OK) {
        ucs_warn("failed to deregister memory handle");
    }

    ucs_free(mem_hndl);
    return status;
}

static int uct_is_cudaipc_copy_mem_type_owned(uct_md_h md, void *addr, size_t length)
{
    int memory_type;
    CUresult cu_ret;

    if (addr == NULL) {
        return 0;
    }

    cu_ret = cuPointerGetAttribute(&memory_type,
                                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)addr);
    if (cu_ret == CUDA_SUCCESS) {
        if (memory_type == CU_MEMORYTYPE_DEVICE) {
            return 1;
        }
    }
    return 0;
}

static ucs_status_t uct_cudaipc_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                        unsigned *num_resources_p)
{
    int num_gpus;
    CUresult cu_ret;

    cu_ret = cuDeviceGetCount(&num_gpus);
    if ((cu_ret != CUDA_SUCCESS) || (num_gpus == 0)) {
        ucs_debug("not found cuda devices");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    return uct_single_md_resource(&uct_cudaipc_copy_md_component, resources_p,
                                  num_resources_p);
}

static void uct_cudaipc_copy_md_close(uct_md_h uct_md)
{
    uct_cudaipc_copy_md_t *md = ucs_derived_of(uct_md, uct_cudaipc_copy_md_t);

    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close              = uct_cudaipc_copy_md_close,
    .query              = uct_cudaipc_copy_md_query,
    .mkey_pack          = uct_cudaipc_copy_mkey_pack,
    .mem_reg            = uct_cudaipc_copy_mem_reg,
    .mem_dereg          = uct_cudaipc_copy_mem_dereg,
    .is_mem_type_owned  = uct_is_cudaipc_copy_mem_type_owned
};

static inline uct_cudaipc_copy_rcache_region_t*
uct_cudaipc_copy_rache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_cudaipc_copy_rcache_region_t, memh);
}

static ucs_status_t
uct_cudaipc_copy_mem_rcache_reg(uct_md_h uct_md, void *address, size_t length,
                                unsigned flags, uct_mem_h *memh_p)
{
    uct_cudaipc_copy_md_t *md = ucs_derived_of(uct_md, uct_cudaipc_copy_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;
    uct_cudaipc_copy_mem_t *memh;

    status = ucs_rcache_get(md->rcache, address, length, PROT_READ|PROT_WRITE,
                            &flags, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    memh = &ucs_derived_of(rregion, uct_cudaipc_copy_rcache_region_t)->memh;
    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t uct_cudaipc_copy_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_cudaipc_copy_md_t *md = ucs_derived_of(uct_md, uct_cudaipc_copy_md_t);
    uct_cudaipc_copy_rcache_region_t *region = uct_cudaipc_copy_rache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t md_rcache_ops = {
    .close              = uct_cudaipc_copy_md_close,
    .query              = uct_cudaipc_copy_md_query,
    .mkey_pack          = uct_cudaipc_copy_mkey_pack,
    .mem_reg            = uct_cudaipc_copy_mem_rcache_reg,
    .mem_dereg          = uct_cudaipc_copy_mem_rcache_dereg,
    .is_mem_type_owned  = uct_is_cudaipc_copy_mem_type_owned
};

static ucs_status_t
uct_cudaipc_copy_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                   void *arg, ucs_rcache_region_t *rregion)
{
    uct_cudaipc_copy_md_t *md = context;
    int *flags = arg;
    uct_cudaipc_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_cudaipc_copy_rcache_region_t);
    return uct_cudaipc_copy_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                             region->super.super.end -
                                             region->super.super.start,
                                             *flags, &region->memh);
}

static void uct_cudaipc_copy_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                                 ucs_rcache_region_t *rregion)
{
    uct_cudaipc_copy_md_t *md = context;
    uct_cudaipc_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_cudaipc_copy_rcache_region_t);
    (void)uct_cudaipc_copy_mem_dereg_internal(&md->super, &region->memh);
}

static void uct_cudaipc_copy_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                                   ucs_rcache_region_t *rregion, char *buf,
                                                   size_t max)
{
    uct_cudaipc_copy_rcache_region_t *region = ucs_derived_of(rregion,
                                                              uct_cudaipc_copy_rcache_region_t);
    uct_cudaipc_copy_mem_t *memh = &region->memh;

    snprintf(buf, max, "d_ptr:%p", (void *) memh->d_ptr);
}

static ucs_rcache_ops_t uct_cudaipc_copy_rcache_ops = {
    .mem_reg     = uct_cudaipc_copy_rcache_mem_reg_cb,
    .mem_dereg   = uct_cudaipc_copy_rcache_mem_dereg_cb,
    .dump_region = uct_cudaipc_copy_rcache_dump_region_cb
};

static ucs_status_t uct_cudaipc_copy_md_open(const char *md_name,
                                             const uct_md_config_t *uct_md_config,
                                             uct_md_h *md_p)
{
    const uct_cudaipc_copy_md_config_t *md_config = ucs_derived_of(uct_md_config,
                                                                   uct_cudaipc_copy_md_config_t);
    ucs_status_t                       status;
    uct_cudaipc_copy_md_t              *md;
    ucs_rcache_params_t                rcache_params;

    md = ucs_malloc(sizeof(uct_cudaipc_copy_md_t), "uct_cudaipc_copy_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_cudaipc_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_cudaipc_copy_md_component;
    md->rcache = NULL;
    md->reg_cost = md_config->uc_reg_cost;

    if (0 && md_config->enable_rcache != UCS_NO) {
        rcache_params.region_struct_size = sizeof(uct_cudaipc_copy_rcache_region_t);
        rcache_params.alignment          = md_config->rcache.alignment;
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = md;
        rcache_params.ops                = &uct_cudaipc_copy_rcache_ops;
        status = ucs_rcache_create(&rcache_params, "cudaipc_copy", NULL, &md->rcache);
        if (status == UCS_OK) {
            md->super.ops         = &md_rcache_ops;
            md->reg_cost.overhead = 0;
            md->reg_cost.growth   = 0;
        } else {
            ucs_assert(md->rcache == NULL);
            if (md_config->enable_rcache == UCS_YES) {
                status = UCS_ERR_IO_ERROR;
                goto err_close_cudaipc;
            } else {
                ucs_debug("could not create registration cache for: %s",
                          ucs_status_string(status));
            }
        }
    }

    *md_p = (uct_md_h) md;
    status = UCS_OK;
out:
    return status;
err_close_cudaipc:
    ucs_free(md);
    goto out;
}

UCT_MD_COMPONENT_DEFINE(uct_cudaipc_copy_md_component, UCT_CUDAIPC_COPY_MD_NAME,
                        uct_cudaipc_copy_query_md_resources, uct_cudaipc_copy_md_open, NULL,
                        uct_cudaipc_copy_rkey_unpack, uct_cudaipc_copy_rkey_release, "CUDAIPC_COPY_",
                        uct_cudaipc_copy_md_config_table, uct_cudaipc_copy_md_config_t);

