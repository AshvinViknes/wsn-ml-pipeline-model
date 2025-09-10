#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include "msg.h"
#include "net/ipv6/addr.h"
#include "net/gnrc.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/ipv6/hdr.h"
#include "net/utils.h"
#include "shell.h"
#include "thread.h"
#include "ztimer.h"

#define MSG_QUEUE_SIZE 8
#define SEND_INTERVAL_USEC (1000000 / 10) // 10 packets/sec
#define MAX_TARGETS 8

static char recv_stack[THREAD_STACKSIZE_DEFAULT];
static char send_stack[THREAD_STACKSIZE_DEFAULT];

static ipv6_addr_t target_addrs[MAX_TARGETS];
static int num_targets = 0;
static netif_t *target_netif = NULL;

static volatile bool sender_running = false;

/* Receiver Thread */
void *recv_handler(void *arg)
{
    (void)arg;
    puts("[recv_thread] started");

    msg_t msg_queue[MSG_QUEUE_SIZE];
    msg_init_queue(msg_queue, MSG_QUEUE_SIZE);

    gnrc_netreg_entry_t server = GNRC_NETREG_ENTRY_INIT_PID(253, thread_getpid());
    gnrc_netreg_register(GNRC_NETTYPE_IPV6, &server);

    while (1)
    {
        msg_t msg;
        msg_receive(&msg);

        if (msg.type == GNRC_NETAPI_MSG_TYPE_RCV)
        {
            gnrc_pktsnip_t *pkt = msg.content.ptr;

            if (pkt->next && pkt->next->next)
            {
                gnrc_netif_hdr_t *nethdr = (gnrc_netif_hdr_t *)pkt->next->next->data;
                int16_t rssi = nethdr->rssi;
                uint8_t lqi = nethdr->lqi;
                printf("%d,%u\n", rssi, lqi);
            }
            gnrc_pktbuf_release(pkt);
        }
    }
    gnrc_netreg_unregister(GNRC_NETTYPE_IPV6, &server);
    return NULL;
}

/* Sender Thread */
void *sender_handler(void *arg)
{
    (void)arg;
    puts("[sender_thread] started");

    while (1)
    {
        ztimer_sleep(ZTIMER_USEC, SEND_INTERVAL_USEC);

        if (num_targets == 0 || !target_netif || !sender_running)
        {
            continue;
        }

        uint64_t now_sec = ztimer_now(ZTIMER_USEC) / 1000000;
        char payload_buf[64];
        snprintf(payload_buf, sizeof(payload_buf), "0,0,%llu", (unsigned long long)now_sec);

        for (int i = 0; i < num_targets; i++)
        {
            gnrc_pktsnip_t *payload = gnrc_pktbuf_add(NULL, payload_buf, strlen(payload_buf), GNRC_NETTYPE_UNDEF);
            if (!payload)
            {
                puts("Payload alloc failed");
                continue;
            }

            gnrc_pktsnip_t *ip = gnrc_ipv6_hdr_build(payload, NULL, &target_addrs[i]);
            if (!ip)
            {
                puts("IPv6 header alloc failed");
                gnrc_pktbuf_release(payload);
                continue;
            }

            ipv6_hdr_t *ip_hdr = ip->data;
            ip_hdr->nh = 253;

            gnrc_pktsnip_t *netif_hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
            if (!netif_hdr)
            {
                puts("Netif header alloc failed");
                gnrc_pktbuf_release(ip);
                continue;
            }

            gnrc_netif_hdr_set_netif(netif_hdr->data, container_of(target_netif, gnrc_netif_t, netif));
            ip = gnrc_pkt_prepend(ip, netif_hdr);

            if (!gnrc_netapi_dispatch_send(GNRC_NETTYPE_IPV6, GNRC_NETREG_DEMUX_CTX_ALL, ip))
            {
                puts("Packet send failed");
                gnrc_pktbuf_release(ip);
            }
        }
    }
    return NULL;
}

/* Shell Command: Add target IPv6 */
int add_target(int argc, char **argv)
{
    if (argc != 2)
    {
        puts("Usage: target_add <IPv6 address>");
        return 1;
    }

    if (num_targets >= MAX_TARGETS)
    {
        puts("Max targets reached");
        return 1;
    }

    ipv6_addr_t addr;
    if (!ipv6_addr_from_str(&addr, argv[1]))
    {
        puts("Invalid IPv6 address");
        return 1;
    }

    target_addrs[num_targets++] = addr;

    if (!target_netif)
    {
        gnrc_netif_t *netif = NULL;
        while ((netif = gnrc_netif_iter(netif)))
        {
            target_netif = &netif->netif;
            break;
        }
    }

    printf("Added target: %s\n", argv[1]);
    return 0;
}

int send_control(int argc, char **argv)
{
    if (argc != 2)
    {
        puts("Usage: send <start|stop>");
        return 1;
    }

    if (strcmp(argv[1], "start") == 0)
    {
        sender_running = true;
        puts("Sender started");
    }
    else if (strcmp(argv[1], "stop") == 0)
    {
        sender_running = false;
        puts("Sender stopped");
    }
    else
    {
        puts("Invalid argument, use start or stop");
        return 1;
    }

    return 0;
}

int status_cmd(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    char addr_str[32];
    printf("Targets (%d):\n", num_targets);
    for (int i = 0; i < num_targets; i++)
    {
        ipv6_addr_to_str(addr_str, &target_addrs[i], sizeof(addr_str));
        printf("  %d: %s\n", i + 1, addr_str);
    }
    printf("Sender running: %s\n", sender_running ? "true" : "false");
    return 0;
}

SHELL_COMMAND(status, "Show sender status and target list", status_cmd);
SHELL_COMMAND(send, "Control sender: send start|stop", send_control);
SHELL_COMMAND(target_add, "Add IPv6 target for sending packets", add_target);

int main(void)
{
    msg_t msg_queue[MSG_QUEUE_SIZE];
    msg_init_queue(msg_queue, MSG_QUEUE_SIZE);

    puts("Packet Logger w/ Timestamp RSSI LQI\n");

    char line_buf[SHELL_DEFAULT_BUFSIZE];

    thread_create(recv_stack, sizeof(recv_stack),
                  THREAD_PRIORITY_MAIN - 1,
                  THREAD_CREATE_STACKTEST,
                  recv_handler, NULL, "recv_thread");

    thread_create(send_stack, sizeof(send_stack),
                  THREAD_PRIORITY_MAIN - 2,
                  THREAD_CREATE_STACKTEST,
                  sender_handler, NULL, "sender_thread");

    shell_run(NULL, line_buf, SHELL_DEFAULT_BUFSIZE);

    return 0;
}
