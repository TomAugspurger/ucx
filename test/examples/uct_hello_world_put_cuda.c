/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "ucx_hello_world.h"
#include <uct/api/uct.h>

#include <assert.h>
#include <ctype.h>

#define CUDA_DEVICE_DEFAULT 0

typedef enum {
    FUNC_PUT_SHORT,
    FUNC_PUT_BCOPY,
    FUNC_PUT_ZCOPY
} func_put_t;

typedef struct {
    int  is_uct_desc;
} recv_desc_t;

typedef struct {
    char               *server_name;
    uint16_t            server_port;
    func_put_t           func_put_type;
    const char         *dev_name;
    const char         *tl_name;
    long                test_strlen;
} cmd_args_t;

typedef struct {
    uct_iface_attr_t    attr;   /* Interface attributes: capabilities and limitations */
    uct_iface_h         iface;  /* Communication interface context */
    uct_md_h            pd;     /* Memory domain */
    uct_md_attr_t       pd_attr;/* Memory domain attributes */
    uct_worker_h        worker; /* Workers represent allocated resources in a communication thread */
} iface_info_t;

/* Helper data type for put_bcopy */
typedef struct {
    char               *data;
    size_t              len;
} put_bcopy_args_t;


static void* desc_holder = NULL;

static char *func_put_t_str(func_put_t func_put_type)
{
    switch (func_put_type) {
    case FUNC_PUT_SHORT:
        return "uct_ep_put_short";
    case FUNC_PUT_BCOPY:
        return "uct_ep_put_bcopy";
    case FUNC_PUT_ZCOPY:
        return "uct_ep_put_zcopy";
    }
    return NULL;
}

static size_t func_put_max_size(func_put_t func_put_type,
                               const uct_iface_attr_t *attr)
{
    switch (func_put_type) {
    case FUNC_PUT_SHORT:
        return attr->cap.put.max_short;
    case FUNC_PUT_BCOPY:
        return attr->cap.put.max_bcopy;
    case FUNC_PUT_ZCOPY:
        return attr->cap.put.max_zcopy;
    }
    return 0;
}

ucs_status_t do_put_short(uct_ep_h ep, const cmd_args_t *cmd_args, char *buf,
                          uint64_t peer_addr, uct_rkey_t rkey)
{
    /* Invoke put on remote endpoint */
    return uct_ep_put_short(ep, buf, cmd_args->test_strlen,
                            peer_addr, rkey);
}

/* Pack callback for put_bcopy */
size_t put_bcopy_data_pack_cb(void *dest, void *arg)
{
    put_bcopy_args_t *bc_args = arg;
    memcpy(dest, bc_args->data, bc_args->len);
    return bc_args->len;
}

/* Executed by the server alone */
ucs_status_t do_put_bcopy(uct_ep_h ep, const cmd_args_t *cmd_args, char *buf,
                          uint64_t peer_addr, uct_rkey_t rkey)
{
    put_bcopy_args_t args;
    ssize_t len;

    args.data = buf;
    args.len  = cmd_args->test_strlen;

    /* Invoke put on remote endpoint */
    len = uct_ep_put_bcopy(ep, put_bcopy_data_pack_cb, &args, peer_addr, rkey);
    /* Negative len is an error code */
    return (len >= 0) ? UCS_OK : len;
}

/* Completion callback for put_zcopy */
void zcopy_completion_cb(uct_completion_t *self, ucs_status_t status)
{
    uct_completion_t *uct_comp = self;
    assert(((*uct_comp).count == 0) && (status == UCS_OK));
}

ucs_status_t do_put_zcopy(uct_ep_h ep, const cmd_args_t *cmd_args, char *buf,
                          uct_mem_h *memh, uint64_t peer_addr, uct_rkey_t rkey,
                          iface_info_t *if_info)
{
    uct_completion_t uct_comp;
    uct_iov_t iov;

    size_t iovcnt       = 1;
    ucs_status_t status = UCS_OK;
    iov.buffer          = buf;
    iov.length          = cmd_args->test_strlen;
    iov.memh            = *memh;
    iov.stride          = 0;
    iov.count           = 1;

    uct_comp.func  = zcopy_completion_cb;
    uct_comp.count = 1;

    if (status == UCS_OK) {
        status = uct_ep_put_zcopy(ep, &iov, iovcnt, peer_addr, rkey, &uct_comp);
        if (status == UCS_INPROGRESS) {
            while (0 != uct_comp.count) {
                /* Explicitly progress outstanding put request */
                uct_worker_progress(if_info->worker);
            }
            status = UCS_OK;
        }
    }
    return status;
}

static void print_strings(const char *label, const char *local_str,
                          const char *remote_str)
{
    fprintf(stdout, "\n\n----- UCT TEST SUCCESS ----\n\n");
    fprintf(stdout, "[%s] %s sent %s", label, local_str, remote_str);
    fprintf(stdout, "\n\n---------------------------\n");
    fflush(stdout);
}

/* init the transport  by its name */
static ucs_status_t init_iface(char *dev_name, char *tl_name,
                               func_put_t func_put_type,
                               iface_info_t *iface_p)
{
    ucs_status_t        status;
    uct_iface_config_t  *config; /* Defines interface configuration options */
    uct_iface_params_t  params;

    params.mode.device.tl_name  = tl_name;
    params.mode.device.dev_name = dev_name;
    params.stats_root           = NULL;
    params.rx_headroom          = sizeof(recv_desc_t);

    UCS_CPU_ZERO(&params.cpu_mask);
    /* Read transport-specific interface configuration */
    status = uct_md_iface_config_read(iface_p->pd, tl_name, NULL, NULL, &config);
    CHKERR_JUMP(UCS_OK != status, "setup iface_config", error_ret);

    /* Open communication interface */
    status = uct_iface_open(iface_p->pd, iface_p->worker, &params, config,
                            &iface_p->iface);
    uct_config_release(config);
    CHKERR_JUMP(UCS_OK != status, "open temporary interface", error_ret);

    /* Get interface attributes */
    status = uct_iface_query(iface_p->iface, &iface_p->attr);
    CHKERR_JUMP(UCS_OK != status, "query iface", error_iface);

    /* Check if current device and transport support required active messages */
    if ((func_put_type == FUNC_PUT_SHORT) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_PUT_SHORT)) {
        return UCS_OK;
    }

    if ((func_put_type == FUNC_PUT_BCOPY) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_PUT_BCOPY)) {
        return UCS_OK;
    }

    if ((func_put_type == FUNC_PUT_ZCOPY) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY)) {
        return UCS_OK;
    }

error_iface:
    uct_iface_close(iface_p->iface);
error_ret:
    return UCS_ERR_UNSUPPORTED;
}

/* Device and transport to be used are determined by minimum latency */
static ucs_status_t dev_tl_lookup(const cmd_args_t *cmd_args,
                                  iface_info_t *iface_p)
{
    uct_md_resource_desc_t  *md_resources; /* Memory domain resource descriptor */
    uct_tl_resource_desc_t  *tl_resources; /*Communication resource descriptor */
    unsigned                num_md_resources; /* Number of protected domain */
    unsigned                num_tl_resources; /* Number of transport resources resource objects created */
    uct_md_config_t         *md_config;
    ucs_status_t            status;
    int                     i;
    int                     j;

    status = uct_query_md_resources(&md_resources, &num_md_resources);
    CHKERR_JUMP(UCS_OK != status, "query for protected domain resources", error_ret);

    fprintf(stderr, "num_md_resources = %d\n", num_md_resources);
    /* List protected domain resources */
    for (i = 0; i < num_md_resources; ++i) {
        fprintf(stderr, "md_name[%d] = %s\n", i, md_resources[i].md_name);
        status = uct_md_config_read(md_resources[i].md_name, NULL, NULL, &md_config);
        CHKERR_JUMP(UCS_OK != status, "read PD config", release_pd);

        status = uct_md_open(md_resources[i].md_name, md_config, &iface_p->pd);
        uct_config_release(md_config);
        CHKERR_JUMP(UCS_OK != status, "open protected domains", release_pd);

        status = uct_md_query_tl_resources(iface_p->pd, &tl_resources, &num_tl_resources);
        CHKERR_JUMP(UCS_OK != status, "query transport resources", close_pd);
        fprintf(stderr, "For md_name[%d] = %s, num_tl_resources = %d\n", i,
                md_resources[i].md_name, num_tl_resources);

        status = uct_md_query(iface_p->pd, &iface_p->pd_attr);
        CHKERR_JUMP(UCS_OK != status, "md attr query error", release_pd);
        fprintf(stderr, "For md_name[%d] = %s, num_tl_resources = %d rkey_packed size = %d\n", i,
                md_resources[i].md_name, num_tl_resources, iface_p->pd_attr.rkey_packed_size);

        /* Go through each available transport and find the proper name */
        for (j = 0; j < num_tl_resources; ++j) {
            fprintf(stderr, "For md_name[%d] = %s, tl_dev_name[%d] = %s, tl_name[%d] = %s \n",
                    i, md_resources[i].md_name,
                    j, tl_resources[j].dev_name,
                    j, tl_resources[j].tl_name);
        }
        uct_release_tl_resource_list(tl_resources);
        uct_md_close(iface_p->pd);
    }

    /* Iterate through protected domain resources */
    for (i = 0; i < num_md_resources; ++i) {
        status = uct_md_config_read(md_resources[i].md_name, NULL, NULL, &md_config);
        CHKERR_JUMP(UCS_OK != status, "read PD config", release_pd);

        status = uct_md_open(md_resources[i].md_name, md_config, &iface_p->pd);
        uct_config_release(md_config);
        CHKERR_JUMP(UCS_OK != status, "open protected domains", release_pd);

        status = uct_md_query(iface_p->pd, &iface_p->pd_attr);
        CHKERR_JUMP(UCS_OK != status, "md attr query error", release_pd);

        status = uct_md_query_tl_resources(iface_p->pd, &tl_resources, &num_tl_resources);
        CHKERR_JUMP(UCS_OK != status, "query transport resources", close_pd);

        /* Go through each available transport and find the proper name */
        for (j = 0; j < num_tl_resources; ++j) {
            if (!strcmp(cmd_args->dev_name, tl_resources[j].dev_name) &&
                !strcmp(cmd_args->tl_name, tl_resources[j].tl_name)) {
                status = init_iface(tl_resources[j].dev_name,
                                    tl_resources[j].tl_name,
                                    cmd_args->func_put_type, iface_p);
                if (UCS_OK == status) {
                    fprintf(stdout, "Using %s with %s.\n",
                            tl_resources[j].dev_name,
                            tl_resources[j].tl_name);
                    fflush(stdout);
                    uct_release_tl_resource_list(tl_resources);
                    goto release_pd;
                }
            }
        }
        uct_release_tl_resource_list(tl_resources);
        uct_md_close(iface_p->pd);
    }

    fprintf(stderr, "No supported (dev/tl) found (%s/%s)\n",
            cmd_args->dev_name, cmd_args->tl_name);
    status = UCS_ERR_UNSUPPORTED;

release_pd:
    uct_release_md_resource_list(md_resources);
error_ret:
    return status;
close_pd:
    uct_md_close(iface_p->pd);
    goto release_pd;
}

int print_err_usage()
{
    const char func_template[] = "  -%c      Select \"%s\" function to send the message%s\n";

    fprintf(stderr, "Usage: uct_hello_world [parameters]\n");
    fprintf(stderr, "UCT hello world client/server example utility\n");
    fprintf(stderr, "\nParameters are:\n");
    fprintf(stderr, func_template, 'i', func_put_t_str(FUNC_PUT_SHORT), " (default)");
    fprintf(stderr, func_template, 'b', func_put_t_str(FUNC_PUT_BCOPY), "");
    fprintf(stderr, func_template, 'z', func_put_t_str(FUNC_PUT_ZCOPY), "");
    fprintf(stderr, "  -d      Select device name\n");
    fprintf(stderr, "  -t      Select transport layer\n");
    fprintf(stderr, "  -n name Set node name or IP address "
            "of the server (required for client and should be ignored "
            "for server)\n");
    fprintf(stderr, "  -p port Set alternative server port (default:13337)\n");
    fprintf(stderr, "  -s size Set test string length (default:16)\n");
    fprintf(stderr, "\n");
    return UCS_ERR_UNSUPPORTED;
}

int parse_cmd(int argc, char * const argv[], cmd_args_t *args)
{
    int c = 0, index = 0;

    assert(args);
    memset(args, 0, sizeof(*args));

    /* Defaults */
    args->server_port   = 13337;
    args->func_put_type  = FUNC_PUT_SHORT;
    args->test_strlen   = 16;

    opterr = 0;
    while ((c = getopt(argc, argv, "ibzd:t:n:p:s:h")) != -1) {
        switch (c) {
        case 'i':
            args->func_put_type = FUNC_PUT_SHORT;
            break;
        case 'b':
            args->func_put_type = FUNC_PUT_BCOPY;
            break;
        case 'z':
            args->func_put_type = FUNC_PUT_ZCOPY;
            break;
        case 'd':
            args->dev_name = optarg;
            break;
        case 't':
            args->tl_name = optarg;
            break;
        case 'n':
            args->server_name = optarg;
            break;
        case 'p':
            args->server_port = atoi(optarg);
            if (args->server_port <= 0) {
                fprintf(stderr, "Wrong server port number %d\n",
                        args->server_port);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case 's':
            args->test_strlen = atol(optarg);
            if (args->test_strlen <= 0) {
                fprintf(stderr, "Wrong string size %ld\n", args->test_strlen);
                return UCS_ERR_UNSUPPORTED;
            }
            break;
        case '?':
            if (optopt == 's') {
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            } else if (isprint (optopt)) {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
        case 'h':
        default:
            return print_err_usage();
        }
    }
    fprintf(stderr, "INFO: UCT_HELLO_WORLD AM function = %s server = %s port = %d\n",
            func_put_t_str(args->func_put_type), args->server_name,
            args->server_port);

    for (index = optind; index < argc; index++) {
        fprintf(stderr, "WARNING: Non-option argument %s\n", argv[index]);
    }

    if (args->dev_name == NULL) {
        fprintf(stderr, "WARNING: device is not set\n");
        return print_err_usage();
    }

    if (args->tl_name == NULL) {
        fprintf(stderr, "WARNING: transport layer is not set\n");
        return print_err_usage();
    }

    return UCS_OK;
}

/* The caller is responsible to free *rbuf */
int sendrecv(int sock, const void *sbuf, size_t slen, void **rbuf)
{
    int ret = 0;
    size_t rlen = 0;
    *rbuf = NULL;

    ret = send(sock, &slen, sizeof(slen), 0);
    if ((ret < 0) || (ret != sizeof(slen))) {
        fprintf(stderr, "failed to send buffer length\n");
        return -1;
    }

    ret = send(sock, sbuf, slen, 0);
    if ((ret < 0) || (ret != slen)) {
        fprintf(stderr, "failed to send buffer\n");
        return -1;
    }

    ret = recv(sock, &rlen, sizeof(rlen), 0);
    if (ret < 0) {
        fprintf(stderr, "failed to receive device address length\n");
        return -1;
    }

    *rbuf = calloc(1, rlen);
    if (!*rbuf) {
        fprintf(stderr, "failed to allocate receive buffer\n");
        return -1;
    }

    ret = recv(sock, *rbuf, rlen, 0);
    if (ret < 0) {
        fprintf(stderr, "failed to receive device address\n");
        return -1;
    }

    return 0;
}

ucs_status_t generate_rkey_buf(void *buf, int len, iface_info_t *if_info,
                               char *rkey_buffer, uct_mem_h *memh_buf)
{
    enum uct_md_mem_flags flags  = UCT_MD_MEM_ACCESS_ALL;
    ucs_status_t          status = UCS_OK;

    status = uct_md_mem_reg(if_info->pd, buf, len, flags, memh_buf);
    CHKERR_JUMP(UCS_OK != status, "memory domain register attempt", out);

    status = uct_md_mkey_pack(if_info->pd, *memh_buf, rkey_buffer);
    CHKERR_JUMP(UCS_OK != status, "memory domain memory key pack", out);

 out:
    return status;
}

ucs_status_t mem_dereg(iface_info_t *if_info, uct_mem_h *memh_buf)
{
    ucs_status_t          status = UCS_OK;

    status = uct_md_mem_dereg(if_info->pd, *memh_buf);
    CHKERR_JUMP(UCS_OK != status, "memory domain buf dereg", out);

 out:
    return status;
}

ucs_status_t generate_rkey_bundle(iface_info_t *if_info, char *rkey_buffer,
                                  uct_rkey_bundle_t *rkey_ob)
{
    int          ret    = 0;
    ucs_status_t status = UCS_OK;

    status = uct_rkey_unpack(rkey_buffer, rkey_ob);
    CHKERR_JUMP(UCS_OK != status, "memory domain rkey unpack", out);

 out:
    return status;
}

void check_correctness(cmd_args_t *cmd_args, char *str)
{
    int ii = 0;
    int errs = 0;
    char *h_str = NULL;
    if (cmd_args->server_name) {
        /* do nothing */
    }
    else {
        h_str = (char *) malloc(sizeof(char) * cmd_args->test_strlen);

        if (cudaSuccess != cudaMemcpy(h_str, str,
                                      cmd_args->test_strlen * sizeof(char),
                                      cudaMemcpyDeviceToHost)) {
            fprintf(stderr, "cudaMemcpy Failed\n");
            return;
        }

        for (ii = 0; ii < cmd_args->test_strlen; ii++) {
            if ('a' != h_str[ii]) errs++;
        }

        free(h_str);
    }
    if (errs) fprintf(stderr, "errors found after put op #errs = %d\n", errs);
}

int cuda_mem_setup(cmd_args_t *cmd_args, void **buffer)
{
    char val   = cmd_args->server_name ? 'a' : 'b';

    if (cudaSuccess != cudaMalloc(buffer, cmd_args->test_strlen)) {
	fprintf(stderr, "cudaMalloc Failed\n");
	return -1;
    }

    if (cudaSuccess != cudaMemset(*buffer, val, cmd_args->test_strlen)) {
	fprintf(stderr, "cudaMalloc Failed\n");
	return -1;
    }

    return 0;
}

int cuda_free(void *buffer)
{
    if (cudaSuccess != cudaFree(buffer)) {
	fprintf(stderr, "cudaFree Failed\n");
	return -1;
    }
    return 0;
}

int cuda_init(cmd_args_t *cmd_args)
{
     int cuda_dev_num = 0;
     int num_cuda_devices = 0;
     struct cudaDeviceProp deviceProp;
     void *tmp;
     cudaError_t cuda_status = cudaSuccess;

     cuda_status = cudaGetDeviceCount(&num_cuda_devices);
     if (cudaSuccess != cuda_status) {
	 fprintf(stderr, "cuda error\n");
	 exit(-1);
     }

     if (num_cuda_devices < 2) {
	 fprintf(stderr, "test needs at least two GPUs\n");
	 return -1;
     }

     cuda_dev_num = (cmd_args->server_name) ? 0 : 1;

     cuda_status = cudaSetDevice(cuda_dev_num);
     if (cudaSuccess != cuda_status) {
	 fprintf(stderr, "cuda set dev error\n");
	 exit(-1);
     }

    return 0;
}

int main(int argc, char **argv)
{
    uct_device_addr_t   *own_dev    = NULL;
    uct_device_addr_t   *peer_dev   = NULL;
    uct_iface_addr_t    *own_iface  = NULL;
    uct_iface_addr_t    *peer_iface = NULL;
    uct_ep_addr_t       *own_ep     = NULL;
    uct_ep_addr_t       *peer_ep    = NULL;
    ucs_status_t        status      = UCS_OK; /* status codes for UCS */
    uct_ep_h            ep;                   /* Remote endpoint */
    ucs_async_context_t *async;               /* Async event context manages
                                                 times and fd notifications */
    cmd_args_t          cmd_args;

    iface_info_t        if_info;
    int                 oob_sock    = -1;     /* OOB connection socket */
    char                *str        = NULL;
    int                 ii          = 0;
    char                own_rkey_buf[128];
    char                *peer_rkey_buf;
    uint64_t            own_addr;
    uint64_t            *peer_addr;
    uct_rkey_t          rkey;
    uct_rkey_bundle_t   rkey_ob;
    uct_mem_h           memh_buf;

    /* Parse the command line */
    if (parse_cmd(argc, argv, &cmd_args)) {
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (cuda_init(&cmd_args)) {
        status = UCS_ERR_UNSUPPORTED;
        fprintf(stderr, "cuda init failed\n");
	goto out;
    }

    /* Initialize context
     * It is better to use different contexts for different workers
     */
    status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD, &async);
    CHKERR_JUMP(UCS_OK != status, "init async context", out);

    /* Create a worker object */
    status = uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &if_info.worker);
    CHKERR_JUMP(UCS_OK != status, "create worker", out_cleanup_async);

    /* Search for the desired transport */
    status = dev_tl_lookup(&cmd_args, &if_info);
    CHKERR_JUMP(UCS_OK != status, "find supported device and transport",
                out_destroy_worker);

    own_dev = (uct_device_addr_t*)calloc(1, if_info.attr.device_addr_len);
    CHKERR_JUMP(NULL == own_dev, "allocate memory for dev addr",
                out_destroy_iface);

    own_iface = (uct_iface_addr_t*)calloc(1, if_info.attr.iface_addr_len);
    CHKERR_JUMP(NULL == own_iface, "allocate memory for if addr",
                out_free_dev_addrs);

    /* Get iface address */
    status = uct_iface_get_address(if_info.iface, own_iface);
    CHKERR_JUMP(UCS_OK != status, "get device address", out_free_if_addrs);

    /* Get device address */
    status = uct_iface_get_device_address(if_info.iface, own_dev);
    CHKERR_JUMP(UCS_OK != status, "get device address", out_free_if_addrs);

    if (cmd_args.server_name) {
        oob_sock = client_connect(cmd_args.server_name, cmd_args.server_port);
        if (oob_sock < 0) {
            goto out_free_if_addrs;
        }
    } else {
        oob_sock = server_connect(cmd_args.server_port);
        if (oob_sock < 0) {
            goto out_free_if_addrs;
        }
    }

    if (if_info.attr.device_addr_len) {
	status = sendrecv(oob_sock, own_dev, if_info.attr.device_addr_len,
			  (void **)&peer_dev);
	CHKERR_JUMP(0 != status, "device exchange", out_free_dev_addrs);
    }

    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
	status = sendrecv(oob_sock, own_iface, if_info.attr.iface_addr_len,
			  (void **)&peer_iface);
	CHKERR_JUMP(0 != status, "iface exchange", out_free_if_addrs);
    }

    status = uct_iface_is_reachable(if_info.iface, NULL, peer_iface);
    CHKERR_JUMP(0 == status, "reach the peer", out_free_if_addrs);

    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        own_ep = (uct_ep_addr_t*)calloc(1, if_info.attr.ep_addr_len);
        CHKERR_JUMP(NULL == own_ep, "allocate memory for ep addrs", out_free_if_addrs);

        /* Create new endpoint */
        status = uct_ep_create(if_info.iface, &ep);
        CHKERR_JUMP(UCS_OK != status, "create endpoint", out_free_ep_addrs);

        /* Get endpoint address */
        status = uct_ep_get_address(ep, own_ep);
        CHKERR_JUMP(UCS_OK != status, "get endpoint address", out_free_ep);

        status = sendrecv(oob_sock, own_ep, if_info.attr.ep_addr_len,
                          (void **)&peer_ep);
        CHKERR_JUMP(0 != status, "EPs exchange", out_free_ep);

        /* Connect endpoint to a remote endpoint */
        status = uct_ep_connect_to_ep(ep, peer_dev, peer_ep);
        barrier(oob_sock);
    } else if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        /* Create an endpoint which is connected to a remote interface */
        status = uct_ep_create_connected(if_info.iface, peer_dev, peer_iface, &ep);
    } else {
        status = UCS_ERR_UNSUPPORTED;
    }
    CHKERR_JUMP(UCS_OK != status, "connect endpoint", out_free_ep);


    if (cmd_args.test_strlen > func_put_max_size(cmd_args.func_put_type,
                                                &if_info.attr)) {
        status = UCS_ERR_UNSUPPORTED;
        fprintf(stderr, "Test string is too long: %ld, max supported: %lu\n",
                cmd_args.test_strlen,
                func_put_max_size(cmd_args.func_put_type, &if_info.attr));
        goto out_free_ep;
    }

    /* allocate and initialize RMA memory */

    if (cuda_mem_setup(&cmd_args, (void **)&str)) {
        status = UCS_ERR_UNSUPPORTED;
        fprintf(stderr, "cuda malloc failed\n");
	goto out_free_ep;
    }

    /* generate access key buffer */
    status = generate_rkey_buf(str, cmd_args.test_strlen, &if_info,
                               own_rkey_buf, &memh_buf);
    CHKERR_JUMP(UCS_OK != status, "rkey buffer creation", out_free_dev_addrs);

    /* Get address for remote addr and rkey exchange */
    status = sendrecv(oob_sock, own_rkey_buf, if_info.pd_attr.rkey_packed_size,
                      (void **)&peer_rkey_buf);
    CHKERR_JUMP(0 != status, "rkey buf exchange", out_free_dev_addrs);

    /* generate access key bundle */
    status = generate_rkey_bundle(&if_info, peer_rkey_buf, &rkey_ob);
    CHKERR_JUMP(UCS_OK != status, "rkey generation", out_free_dev_addrs);

    barrier(oob_sock);

    own_addr = (uint64_t) str;

    status = sendrecv(oob_sock, &own_addr, sizeof(uint64_t),
                      (void **)&peer_addr);
    CHKERR_JUMP(0 != status, "buf address exchange", out_free_dev_addrs);

    /* At this point rkey and remote address is available for put operations */

    if (cmd_args.server_name) {

        /* Invoke put on remote endpoint */
        if (cmd_args.func_put_type == FUNC_PUT_ZCOPY) {
            status = do_put_zcopy(ep, &cmd_args, str, &memh_buf, *peer_addr,
                                  rkey_ob.rkey, &if_info);
        }
        CHKERR_JUMP(UCS_OK != status, "put op", out_free_ep);
    } else {
        if (cmd_args.func_put_type == FUNC_PUT_ZCOPY) {
            /* do nothing */
        }
    }

    barrier(oob_sock);

    check_correctness(&cmd_args, str);

    uct_rkey_release(&rkey_ob);
    mem_dereg(&if_info, &memh_buf);

    if (cuda_free(str)) {
        status = UCS_ERR_UNSUPPORTED;
        fprintf(stderr, "cuda free failed\n");
	goto out_free_ep;
    }

    barrier(oob_sock);

    close(oob_sock);

out_free_ep:
    uct_ep_destroy(ep);
out_free_ep_addrs:
    free(own_ep);
    free(peer_ep);
out_free_if_addrs:
    free(own_iface);
    free(peer_iface);
out_free_dev_addrs:
    free(own_dev);
    free(peer_dev);
out_destroy_iface:
    uct_iface_close(if_info.iface);
    uct_md_close(if_info.pd);
out_destroy_worker:
    uct_worker_destroy(if_info.worker);
out_cleanup_async:
    ucs_async_context_destroy(async);
out:
    return status == UCS_ERR_UNSUPPORTED ? UCS_OK : status;
}
