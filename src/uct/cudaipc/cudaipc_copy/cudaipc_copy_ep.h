/*
 *  Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *  See COPYRIGHT for license information
 */

#ifndef UCT_CUDAIPC_COPY_EP_H
#define UCT_CUDAIPC_COPY_EP_H

#include "cudaipc_copy_md.h"
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>

#include <ucs/datastruct/sglib.h>
#include <ucs/datastruct/sglib_wrapper.h>

#define UCT_CUDAIPC_COPY_HASH_SIZE 256

typedef struct uct_cudaipc_copy_rem_seg  uct_cudaipc_copy_rem_seg_t;

typedef struct uct_cudaipc_copy_rem_seg {
    uct_cudaipc_copy_rem_seg_t *next;
    CUipcMemHandle             ph;         /* Memory handle of GPU memory */
    CUdeviceptr                d_bptr;     /* Allocation base address */
    size_t                     b_len;      /* Allocation size */
    int                        dev_num;    /* GPU Device number */
} uct_cudaipc_copy_rem_seg_t;

typedef struct uct_cudaipc_copy_ep {
    uct_base_ep_t              super;
    uct_cudaipc_copy_rem_seg_t *rem_segments_hash[UCT_CUDAIPC_COPY_HASH_SIZE];
    struct uct_cudaipc_copy_ep *next;
} uct_cudaipc_copy_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_cudaipc_copy_ep_t, uct_ep_t, uct_iface_t*,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cudaipc_copy_ep_t, uct_ep_t);

ucs_status_t uct_cudaipc_copy_ep_get_zcopy(uct_ep_h tl_ep,
                                           const uct_iov_t *iov, size_t iovcnt,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp);

ucs_status_t uct_cudaipc_copy_ep_put_zcopy(uct_ep_h tl_ep,
                                           const uct_iov_t *iov, size_t iovcnt,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp);

/* Not sure how good a hash function this is */
static inline uint64_t
uct_cudaipc_copy_rem_seg_hash(uct_cudaipc_copy_rem_seg_t *seg)
{
    int      i;
    uint64_t hash_val = 7;
    for (i = 0; i < sizeof(seg->ph); i++) {
        hash_val = hash_val*31 + seg->ph.reserved[i];
    }
    return (uint64_t) (hash_val % UCT_CUDAIPC_COPY_HASH_SIZE);
}

static inline uint64_t
uct_cudaipc_copy_rem_seg_compare(uct_cudaipc_copy_rem_seg_t *seg1,
                                 uct_cudaipc_copy_rem_seg_t *seg2)
{
    return (uint64_t) (strncmp(seg1->ph.reserved, seg2->ph.reserved,
                                    sizeof(CUipcMemHandle)));
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_cudaipc_copy_rem_seg_t,
                             uct_cudaipc_copy_rem_seg_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_cudaipc_copy_rem_seg_t,
                                         UCT_CUDAIPC_COPY_HASH_SIZE,
                                         uct_cudaipc_copy_rem_seg_hash)
#endif
