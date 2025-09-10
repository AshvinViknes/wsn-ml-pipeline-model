APPLICATION = pkt_logger

BOARD ?= feather-nrf52840-sense

UF2_SOFTDEV := DROP

RIOTBASE ?= $(CURDIR)/../../RIOT

# Required modules
USEMODULE += gnrc
USEMODULE += gnrc_ipv6
USEMODULE += gnrc_ipv6_default
USEMODULE += gnrc_udp
USEMODULE += gnrc_icmpv6_echo
USEMODULE += auto_init_gnrc_netif
USEMODULE += gnrc_netif_single
USEMODULE += netdev_default
USEMODULE += shell
USEMODULE += shell_cmds_default
USEMODULE += ztimer_usec
USEMODULE += od
USEMODULE += netutils

DEVELHELP ?= 1
QUIET ?= 1

include $(RIOTBASE)/Makefile.include
