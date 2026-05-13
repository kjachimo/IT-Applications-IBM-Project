import socket
import sys
from typing import Optional

import torcs_utils

from driver_action import DriverAction
from server_state import ServerState

data_size = 2**17


class SimpleClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        sid: str = "SCR",
        max_episodes: int = 1,
        trackname: str = "corkscrew",
        stage: int = 3,
        debug: bool = False,
        max_steps: int = 100000,
        vision: bool = False,
        reconnect_attempts: int = 5,
        launch_on_fail: bool = True,
        torcs_process: Optional[object] = None,
    ):
        self.vision = vision

        self.host = host
        self.port = port
        self.sid = sid
        self.maxEpisodes = max_episodes  # Maximum number of learning episodes
        self.trackname = trackname
        self.stage = stage  # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown
        self.debug = debug
        self.maxSteps = max_steps  # 50steps/second
        self.reconnect_attempts = reconnect_attempts
        self.launch_on_fail = launch_on_fail
        self._torcs_process = torcs_process

        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self) -> None:
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            raise RuntimeError("Error: Could not create socket...")
        self.so.settimeout(1)
        a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
        initmsg = f"{self.sid}(init {a})"

        attempts_left = self.reconnect_attempts
        while True:
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as exc:
                raise RuntimeError(f"Error sending init message: {exc}")

            sockdata = str()
            try:
                sockdata, _addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode("utf-8")
            except socket.error:
                print(f"Waiting for server on {self.port}............")
                print(f"Count Down : {attempts_left}")
                if attempts_left <= 0:
                    if self.launch_on_fail:
                        print("relaunching torcs")
                        self._torcs_process = torcs_utils.reset_torcs(
                            vision=self.vision,
                            process=self._torcs_process,
                        )
                        attempts_left = self.reconnect_attempts
                    else:
                        raise RuntimeError("TORCS server did not respond.")
                else:
                    attempts_left -= 1
                continue

            if "***identified***" in sockdata:
                print(f"Client connected on {self.port}..............")
                break

    def get_servers_input(self) -> None:
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

    def respond_to_server(self) -> None:
        if not self.so:
            return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            raise RuntimeError(
                "Error sending to server: %s Message %s" % (emsg[1], str(emsg[0]))
            )
        if self.debug:
            print(self.R.fancyout())

    def shutdown(self) -> None:
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
