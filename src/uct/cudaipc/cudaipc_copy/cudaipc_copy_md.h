/*
 *  Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *  See COPYRIGHT for license information
 */

#ifndef UCT_CUDAIPC_COPY_MD_H
#define UCT_CUDAIPC_COPY_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/rcache.h>
#include <cuda.h>

#define UCT_CUDAIPC_COPY_MD_NAME   "cudaipc"
#define CUDA_ERR_STRLEN            512
#define UCT_CUDAIPC_MAX_ALLOC_SZ   (1 << 24)

extern uct_md_component_t uct_cudaipc_copy_md_component;
const char                *cu_err_str;
CUcontext                 pctx;

/**
 * @brief cudaipc_copy MD descriptor
 */
typedef struct uct_cudaipc_copy_md {
    struct uct_md       super;    /* Domain info */
    ucs_rcache_t        *rcache;  /* Registration cache (can be NULL) */
    uct_linear_growth_t reg_cost; /* Memory registration cost */
} uct_cudaipc_copy_md_t;

/**
 * @brief cudaipc_copy domain configuration.
 */
typedef struct uct_cudaipc_copy_md_config {
    uct_md_config_t        super;
    int                    enable_rcache;/* Enable registration cache */
    uct_md_rcache_config_t rcache;       /* Registration cache config */
    uct_linear_growth_t    uc_reg_cost;  /* Memory registration cost estimation
                                            without using the cache */
} uct_cudaipc_copy_md_config_t;

/**
 * @brief cudaipc copy mem handle
 */
typedef struct uct_cudaipc_copy_mem {
    CUipcMemHandle ph;         /* Memory handle of GPU memory */
    CUdeviceptr    d_ptr;      /* GPU address */
    CUdeviceptr    d_bptr;     /* Allocation base address */
    size_t         b_len;      /* Allocation size */
    int            dev_num;    /* GPU Device number */
    size_t         reg_size;   /* Size of mapping */
} uct_cudaipc_copy_mem_t;

/**
 * @brief cudaipc copy packed and remote key for put/get
 */
typedef struct uct_cudaipc_copy_key {
    CUipcMemHandle ph;           /* Memory handle of GPU memory */
    CUdeviceptr    d_rem_ptr;    /* GPU address */
    CUdeviceptr    d_rem_bptr;   /* Allocation base address */
    size_t         b_rem_len;    /* Allocation size */
    CUdeviceptr    d_mapped_ptr; /* Mapped GPU address */
    int            dev_num;      /* GPU Device number */
} uct_cudaipc_copy_key_t;

/**
 * cudaipc memory region in the registration cache.
 */
typedef struct uct_cudaipc_copy_rcache_region {
    ucs_rcache_region_t    super;
    uct_cudaipc_copy_mem_t memh;  /*  mr exposed to the user as the memh */
} uct_cudaipc_copy_rcache_region_t;

#endif
