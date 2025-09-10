# **Transmission_Code — Data Collection on RIOT**

Minimal RIOT app to **transmit test packets** and **log link-layer RSSI/LQI** on the receiver for the WSN project.

## **Features**

- Sender thread: 10 packets/sec to one or more IPv6 targets.
- Receiver thread: prints **rssi,lqi** per received packet.
- Shell commands: target_add, send start|stop, status.

## **Build & Run (example)**

```bash
make all
make flash
make term | tee node_A_from_B_env_1.txt
```

## **Shell Commands**

1) status: Shows the current target list and sending state.
2) send start: Starts the sender thread. Packets will now be transmitted to all added targets. Use this after you have added target addresses.
3) send stop: Stops the sender thread. Useful for pausing transmission without clearing the target list.

## Usage

1. Add one or more targets with `target_add <IPv6>`.
2. Check with `status` (should show targets, but sender not running yet).
3. Run `send start` to begin sending packets (all targets get packets from the beginning).
4. Optionally run `send stop` to pause.
5. Run `send start` again to resume.

## **Receiver Example**

On the receiving node, run make term to open the serial console.

Each line after the timestamp shows **RSSI,LQI** values extracted from received packets.

Example output:

```bash
/home/server/WSN-Projects/riot-exercises/RIOT/dist/tools/pyterm/pyterm -p "/dev/ttyACM0" -b "115200" -ln "/tmp/pyterm-server" -rn "2025-07-05_16.59.21-pkt_logger-feather-nrf52840-sense"  
2025-07-05 16:59:21,504 # Connect to serial port /dev/ttyACM0
Welcome to pyterm!
Type '/exit' to exit.
2025-07-05 17:06:02,459 # -83,40
2025-07-05 17:06:02,559 # -85,32
2025-07-05 17:06:02,663 # -88,20
```

