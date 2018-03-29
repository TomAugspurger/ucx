/*
 *  Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
 *  See COPYRIGHT for license information
 */

#ifndef UCT_CUDAIPC_COPY_IFACE_H
#define UCT_CUDAIPC_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda.h>

#define UCT_CUDAIPC_COPY_TL_NAME "cudaipc"
#define UCT_CUDAIPC_DEV_NAME     "cudaipc"
#define UCT_CUDAIPC_MAX_PEERS    16

typedef struct uct_cudaipc_copy_iface {
    uct_base_iface_t super;
    ucs_mpool_t      cudaipc_event_desc;
    ucs_queue_head_t outstanding_d2d_cudaipc_event_q;
    ucs_queue_head_t outstanding_h2d_cudaipc_event_q;
    ucs_queue_head_t outstanding_d2h_cudaipc_event_q;
    int              device_count;
    int              streams_initialized;
    CUstream         stream_d2d[UCT_CUDAIPC_MAX_PEERS];
    CUstream         stream_d2h[UCT_CUDAIPC_MAX_PEERS];
    CUstream         stream_h2d[UCT_CUDAIPC_MAX_PEERS];
    int              cudaipc_p2p_map[UCT_CUDAIPC_MAX_PEERS][UCT_CUDAIPC_MAX_PEERS];
    struct {
        unsigned     max_poll;
    } config;
} uct_cudaipc_copy_iface_t;

typedef struct uct_cudaipc_copy_iface_config {
    uct_iface_config_t super;
    unsigned           max_poll;
} uct_cudaipc_copy_iface_config_t;

typedef struct uct_cudaipc_copy_event_desc {
    CUevent           event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cudaipc_copy_event_desc_t;

ucs_status_t uct_cudaipc_copy_iface_init_streams(uct_cudaipc_copy_iface_t *iface);

#endif
