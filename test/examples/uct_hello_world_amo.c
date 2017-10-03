/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucx_hello_world.h"

#include <uct/api/uct.h>

#include <assert.h>
#include <ctype.h>

#define SERVER_VAL 63
#define CLIENT_VAL 64

typedef enum {
    FUNC_ATOMIC_ADD32,
    FUNC_ATOMIC_ADD64,
    FUNC_ATOMIC_FADD32,
    FUNC_ATOMIC_FADD64,
    FUNC_ATOMIC_SWAP32,
    FUNC_ATOMIC_SWAP64,
    FUNC_ATOMIC_CSWAP32,
    FUNC_ATOMIC_CSWAP64
} func_atomic_t;

typedef struct {
    int  is_uct_desc;
} recv_desc_t;

typedef struct {
    char               *server_name;
    uint16_t            server_port;
    char               *atomic_name;
    func_atomic_t       func_atomic_type;
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

static void* desc_holder = NULL;

static char *func_atomic_t_str(func_atomic_t func_atomic_type)
{
    switch (func_atomic_type) {
    case FUNC_ATOMIC_ADD32:
        return "uct_ep_atomic_add32";
    case FUNC_ATOMIC_ADD64:
        return "uct_ep_atomic_add64";
    case FUNC_ATOMIC_FADD32:
        return "uct_ep_atomic_fadd32";
    case FUNC_ATOMIC_FADD64:
        return "uct_ep_atomic_fadd64";
    case FUNC_ATOMIC_SWAP32:
        return "uct_ep_atomic_swap32";
    case FUNC_ATOMIC_SWAP64:
        return "uct_ep_atomic_swap64";
    case FUNC_ATOMIC_CSWAP32:
        return "uct_ep_atomic_cswap32";
    case FUNC_ATOMIC_CSWAP64:
        return "uct_ep_atomic_cswap64";
    }
    return NULL;
}

/* Completion callback for atomic_zcopy */
void zcopy_completion_cb(uct_completion_t *self, ucs_status_t status)
{
    uct_completion_t *uct_comp = self;
    assert(((*uct_comp).count == 0) && (status == UCS_OK));
}

/* init the transport  by its name */
static ucs_status_t init_iface(char *dev_name, char *tl_name,
                               func_atomic_t func_atomic_type,
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
    status = uct_iface_config_read(tl_name, NULL, NULL, &config);
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
    if ((func_atomic_type == FUNC_ATOMIC_ADD32) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_ADD32)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_ADD64) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_ADD64)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_FADD32) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_FADD32)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_FADD64) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_FADD64)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_SWAP32) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_SWAP32)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_SWAP64) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_SWAP64)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_CSWAP32) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CSWAP32)) {
        return UCS_OK;
    }
    if ((func_atomic_type == FUNC_ATOMIC_CSWAP64) &&
        (iface_p->attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CSWAP64)) {
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
                                    cmd_args->func_atomic_type, iface_p);
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
    fprintf(stderr, "  -a      Select atomic name [add32|add64|fadd32|fadd64|swap32|swap64|cswap32|cswap64]\n");
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
    args->func_atomic_type  = FUNC_ATOMIC_ADD32;
    args->test_strlen   = sizeof(uint32_t);

    opterr = 0;
    while ((c = getopt(argc, argv, "d:t:n:a:p:h")) != -1) {
        switch (c) {
        case 'a':
            args->atomic_name = optarg;
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
        case '?':
            if (isprint (optopt)) {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            } else {
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            }
        case 'h':
        default:
            return print_err_usage();
        }
    }

    if (args->atomic_name != NULL) {
        if (0 == strcmp(args->atomic_name, "add32")) {
            args->func_atomic_type = FUNC_ATOMIC_ADD32;
            args->test_strlen   = sizeof(uint32_t);
        }
        else if (0 == strcmp(args->atomic_name, "add64")) {
            args->func_atomic_type = FUNC_ATOMIC_ADD64;
            args->test_strlen   = sizeof(uint64_t);
        }
        else if (0 == strcmp(args->atomic_name, "fadd32")) {
            args->func_atomic_type = FUNC_ATOMIC_FADD32;
            args->test_strlen   = sizeof(uint32_t);
        }
        else if (0 == strcmp(args->atomic_name, "fadd64")) {
            args->func_atomic_type = FUNC_ATOMIC_FADD64;
            args->test_strlen   = sizeof(uint64_t);
        }
        else if (0 == strcmp(args->atomic_name, "swap32")) {
            args->func_atomic_type = FUNC_ATOMIC_SWAP32;
            args->test_strlen   = sizeof(uint32_t);
        }
        else if (0 == strcmp(args->atomic_name, "swap64")) {
            args->func_atomic_type = FUNC_ATOMIC_SWAP64;
            args->test_strlen   = sizeof(uint64_t);
        }
        else if (0 == strcmp(args->atomic_name, "cswap32")) {
            args->func_atomic_type = FUNC_ATOMIC_CSWAP32;
            args->test_strlen   = sizeof(uint32_t);
        }
        else if (0 == strcmp(args->atomic_name, "cswap64")) {
            args->func_atomic_type = FUNC_ATOMIC_CSWAP64;
            args->test_strlen   = sizeof(uint64_t);
        }
        else {
            fprintf(stderr, "WARNING: atomic name option invalid\n");
            return print_err_usage();
        }
    }

    fprintf(stderr, "INFO: UCT_HELLO_WORLD AM function = %s server = %s port = %d\n",
            func_atomic_t_str(args->func_atomic_type), args->server_name,
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

int main(int argc, char **argv)
{
    uct_device_addr_t   *own_dev;
    uct_device_addr_t   *peer_dev   = NULL;
    uct_iface_addr_t    *own_iface;
    uct_iface_addr_t    *peer_iface = NULL;
    uct_ep_addr_t       *own_ep;
    uct_ep_addr_t       *peer_ep    = NULL;
    ucs_status_t        status      = UCS_OK; /* status codes for UCS */
    uct_ep_h            ep;                   /* Remote endpoint */
    ucs_async_context_t *async;               /* Async event context manages
                                                 times and fd notifications */
    cmd_args_t          cmd_args;

    iface_info_t        if_info;
    int                 oob_sock    = -1;     /* OOB connection socket */
    char                *str          = NULL;
    uint32_t            *str32        = NULL;
    uint64_t            *str64        = NULL;
    uint32_t            result32      = 0;
    uint64_t            result64      = 0;
    uint32_t            compare32     = 0;
    uint64_t            compare64     = 0;
    int                 ii            = 0;
    char                own_rkey_buf[128];
    char                *peer_rkey_buf;
    uint64_t            own_addr;
    uint64_t            *peer_addr;
    uct_rkey_t          rkey;
    uct_rkey_bundle_t   rkey_ob;
    uct_mem_h           memh_buf;
    uct_completion_t    uct_comp;

    /* Parse the command line */
    if (parse_cmd(argc, argv, &cmd_args)) {
        status = UCS_ERR_INVALID_PARAM;
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

    status = sendrecv(oob_sock, own_dev, if_info.attr.device_addr_len,
                      (void **)&peer_dev);
    CHKERR_JUMP(0 != status, "device exchange", out_free_dev_addrs);

    status = uct_iface_is_reachable(if_info.iface, peer_dev, NULL);
    CHKERR_JUMP(0 == status, "reach the peer", out_free_if_addrs);

    /* Get interface address */
    if (if_info.attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        status = uct_iface_get_address(if_info.iface, own_iface);
        CHKERR_JUMP(UCS_OK != status, "get interface address", out_free_if_addrs);

        status = sendrecv(oob_sock, own_iface, if_info.attr.iface_addr_len,
                          (void **)&peer_iface);
        CHKERR_JUMP(0 != status, "ifaces exchange", out_free_if_addrs);
    }

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

    /* allocate and initialize RMA memory */
    if (cmd_args.test_strlen == sizeof(uint64_t)) {
        str64 = (uint64_t *) malloc(sizeof(uint64_t));
        *str64 = cmd_args.server_name ? CLIENT_VAL : SERVER_VAL;
        compare64 = cmd_args.server_name ? SERVER_VAL : CLIENT_VAL;
        str = (char *) str64;
    }
    else {
        str32 = (uint32_t *) malloc(sizeof(uint32_t));
        *str32 = cmd_args.server_name ? CLIENT_VAL : SERVER_VAL;
        compare32 = cmd_args.server_name ? SERVER_VAL : CLIENT_VAL;
        str = (char *) str32;
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

    /* At this point rkey and remote address is available for atomic operations */

    uct_comp.func  = zcopy_completion_cb;
    uct_comp.count = 1;

    if (cmd_args.server_name) {
        /* Invoke atomic on remote endpoint */
        if (cmd_args.func_atomic_type == FUNC_ATOMIC_ADD32) {
            status = uct_ep_atomic_add32(ep, (uint32_t) *str, *peer_addr,
                                         rkey_ob.rkey);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_ADD64) {
            status = uct_ep_atomic_add64(ep, (uint64_t) *str, *peer_addr,
                                         rkey_ob.rkey);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_FADD32) {
            status = uct_ep_atomic_fadd32(ep, (uint32_t) *str, *peer_addr,
                                          rkey_ob.rkey, &result32, &uct_comp);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_FADD64) {
            status = uct_ep_atomic_fadd64(ep, (uint64_t) *str, *peer_addr,
                                          rkey_ob.rkey, &result64, &uct_comp);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_SWAP32) {
            status = uct_ep_atomic_swap32(ep, (uint32_t) *str, *peer_addr,
                                          rkey_ob.rkey, &result32, &uct_comp);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_SWAP64) {
            status = uct_ep_atomic_swap64(ep, (uint64_t) *str, *peer_addr,
                                          rkey_ob.rkey, &result64, &uct_comp);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_CSWAP32) {
            status = uct_ep_atomic_cswap32(ep, compare32, (uint32_t) *str,
                                           *peer_addr, rkey_ob.rkey,
                                           &result32, &uct_comp);
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_CSWAP64) {
            status = uct_ep_atomic_cswap64(ep, compare64, (uint64_t) *str,
                                           *peer_addr, rkey_ob.rkey,
                                           &result64, &uct_comp);
        }
        if (status == UCS_INPROGRESS) {
            while (0 != uct_comp.count) {
                /* Explicitly progress outstanding atomic request */
                uct_worker_progress(if_info.worker);
            }
            status = UCS_OK;
        }
        CHKERR_JUMP(UCS_OK != status, "atomic op", out_free_ep);
    } else {
        if (cmd_args.test_strlen == sizeof(uint64_t)) {
            while ((uint64_t) *str == (SERVER_VAL)) {
                /* do nothing */
            }
        }
        else {
            while ((uint32_t) *str == (SERVER_VAL)) {
                /* do nothing */
            }
        }
    }

    barrier(oob_sock);

    /* check_correctness */
    if (cmd_args.server_name) {
        if (cmd_args.func_atomic_type == FUNC_ATOMIC_FADD32) {
            if (result32 == SERVER_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", SERVER_VAL, result32);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_FADD64) {
            if (result64 == SERVER_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", SERVER_VAL, result64);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_SWAP32) {
            if (result32 == SERVER_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", SERVER_VAL, result32);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_SWAP64) {
            if (result64 == SERVER_VAL)  {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", SERVER_VAL, result64);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_CSWAP32) {
            if (result32 == SERVER_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", SERVER_VAL, result32);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_CSWAP64) {
            if (result64 == SERVER_VAL)  {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", SERVER_VAL, result64);
            }
        }
    }
    else {
        if (cmd_args.func_atomic_type == FUNC_ATOMIC_ADD32) {
            if ((uint32_t) *str == (SERVER_VAL + CLIENT_VAL)) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", (SERVER_VAL + CLIENT_VAL), (uint32_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_ADD64) {
            if ((uint64_t) *str == (SERVER_VAL + CLIENT_VAL)) {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", (SERVER_VAL + CLIENT_VAL), (uint64_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_FADD32) {
            if ((uint32_t) *str == (SERVER_VAL + CLIENT_VAL)) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", (SERVER_VAL + CLIENT_VAL), (uint32_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_FADD64) {
            if ((uint64_t) *str == (SERVER_VAL + CLIENT_VAL)) {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", (SERVER_VAL + CLIENT_VAL), (uint64_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_SWAP32) {
            if ((uint32_t) *str == CLIENT_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", (CLIENT_VAL), (uint32_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_SWAP64) {
            if ((uint64_t) *str == CLIENT_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", (CLIENT_VAL), (uint64_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_CSWAP32) {
            if ((uint32_t) *str == CLIENT_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %d\n", (CLIENT_VAL), (uint32_t) *str);
            }
        }
        else if (cmd_args.func_atomic_type == FUNC_ATOMIC_CSWAP64) {
            if ((uint64_t) *str == CLIENT_VAL) {
            }
            else {
                fprintf(stderr, "expecting %d, found %ld\n", (CLIENT_VAL), (uint64_t) *str);
            }
        }
    }

    uct_rkey_release(&rkey_ob);
    mem_dereg(&if_info, &memh_buf);
    free(str);

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
