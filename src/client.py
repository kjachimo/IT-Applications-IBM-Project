import socket
import sys
import getopt
import os
import time

from .server_state import ServerState
from .driver_action import DriverAction

data_size = 2**17

ophelp = "Options:\n"
ophelp += " --host, -H <host>    TORCS server host. [localhost]\n"
ophelp += " --port, -p <port>    TORCS port. [3001]\n"
ophelp += " --id, -i <id>        ID for server. [SCR]\n"
ophelp += " --steps, -m <#>      Maximum simulation steps. 1 sec ~ 50 steps. [100000]\n"
ophelp += " --episodes, -e <#>   Maximum learning episodes. [1]\n"
ophelp += (
    " --track, -t <track>  Your name for this track. Used for learning. [unknown]\n"
)
ophelp += " --stage, -s <#>      0=warm up, 1=qualifying, 2=race, 3=unknown. [3]\n"
ophelp += " --debug, -d          Output full telemetry.\n"
ophelp += " --help, -h           Show this help.\n"
ophelp += " --version, -v        Show current version."
usage = "Usage: %s [ophelp [optargs]] \n" % sys.argv[0]
usage = usage + ophelp

version = "20130505-2"


class Client:
    def __init__(
        self, H=None, p=None, i=None, e=None, t=None, s=None, d=None, vision=False
    ):
        self.vision = vision

        self.host = "localhost"
        self.port = 3001
        self.sid = "SCR"
        self.maxEpisodes = 1  # "Maximum number of learning episodes to perform"
        self.trackname = "unknown"
        self.stage = 3  # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug = False
        self.maxSteps = 100000  # 50steps/second
        self.parse_the_command_line()
        if H:
            self.host = H
        if p:
            self.port = p
        if i:
            self.sid = i
        if e:
            self.maxEpisodes = e
        if t:
            self.trackname = t
        if s:
            self.stage = s
        if d:
            self.debug = d
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print("Error: Could not create socket...")
            sys.exit(-1)
        self.so.settimeout(1)

        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg = "%s(init %s)" % (self.sid, a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    print("relaunch torcs")
                    os.system("pkill torcs")
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system("torcs -nofuel -nodamage -nolaptime &")
                    else:
                        os.system("torcs -nofuel -nodamage -nolaptime -vision &")

                    time.sleep(1.0)
                    os.system("sh autostart.sh")
                    n_fail = 5
                n_fail -= 1

            identify = "***identified***"
            if identify in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(
                sys.argv[1:],
                "H:p:i:m:e:t:s:dhv",
                [
                    "host=",
                    "port=",
                    "id=",
                    "steps=",
                    "episodes=",
                    "track=",
                    "stage=",
                    "debug",
                    "help",
                    "version",
                ],
            )
        except getopt.error as why:
            print("getopt error: %s\n%s" % (why, usage))
            sys.exit(-1)
        try:
            for opt in opts:
                if opt[0] == "-h" or opt[0] == "--help":
                    print(usage)
                    sys.exit(0)
                if opt[0] == "-d" or opt[0] == "--debug":
                    self.debug = True
                if opt[0] == "-H" or opt[0] == "--host":
                    self.host = opt[1]
                if opt[0] == "-i" or opt[0] == "--id":
                    self.sid = opt[1]
                if opt[0] == "-t" or opt[0] == "--track":
                    self.trackname = opt[1]
                if opt[0] == "-s" or opt[0] == "--stage":
                    self.stage = int(opt[1])
                if opt[0] == "-p" or opt[0] == "--port":
                    self.port = int(opt[1])
                if opt[0] == "-e" or opt[0] == "--episodes":
                    self.maxEpisodes = int(opt[1])
                if opt[0] == "-m" or opt[0] == "--steps":
                    self.maxSteps = int(opt[1])
                if opt[0] == "-v" or opt[0] == "--version":
                    print("%s %s" % (sys.argv[0], version))
                    sys.exit(0)
        except ValueError as why:
            print(
                "Bad parameter '%s' for option %s: %s\n%s"
                % (opt[1], opt[0], why, usage)
            )
            sys.exit(-1)
        if len(args) > 0:
            print("Superflous input? %s\n%s" % (", ".join(args), usage))
            sys.exit(-1)

    def get_servers_input(self):
        """Server's input is stored in a ServerState object"""
        if not self.so:
            return
        sockdata = str()

        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                print(".", end=" ")
            if "***identified***" in sockdata:
                print("Client connected on %d.............." % self.port)
                continue
            elif "***shutdown***" in sockdata:
                print(
                    (
                        (
                            "Server has stopped the race on %d. "
                            + "You were in %d place."
                        )
                        % (self.port, self.S.d["racePos"])
                    )
                )
                self.shutdown()
                return
            elif "***restart***" in sockdata:
                print("Server has restarted the race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata:  # Empty?
                continue  # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")  # Clear for steady output.
                    print(self.S)
                break  # Can now return from this function.

    def respond_to_server(self):
        if not self.so:
            return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        if self.debug:
            print(self.R.fancyout())

    def shutdown(self):
        if not self.so:
            return
        print(
            (
                "Race terminated or %d steps elapsed. Shutting down %d."
                % (self.maxSteps, self.port)
            )
        )
        self.so.close()
        self.so = None
