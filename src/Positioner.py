import zmq
import threading
import json
import numpy as np

class PositionerValues(object):
    def __init__(self, arr):
        self.values = arr

    def get_x_positions(self):
        return np.asarray([pos.x for pos in self.values])

    def get_y_positions(self):
        return np.asarray([pos.y for pos in self.values])


class PositionerValue(object):
    """Class that contains all the positioner data, ie x,y,z, time and rotation matrix

    Args:
        object (_type_): _description_
    """

    def __init__(self, timestamp, x, y, z, rotation_matrix=None):
        self.t = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.rotation_matrix = rotation_matrix

    @staticmethod
    def json_decoder(obj):
        if obj is not None:
            return PositionerValue(
                timestamp=obj["t"],
                x=obj["x"],
                y=obj["y"],
                z=obj["z"],
                # TODO add rotation matrix
            )

    def __eq__(self, other: object) -> bool:
        # TODO check when will be say the vals are equal
        if not isinstance(other, PositionerValue):
            return False
        return self.t == other.t

    def __str__(self):
        return f"({self.x},{self.y},{self.z}) @ t={self.t}s"


class PositionerClient:
    def __init__(self, config: dict) -> None:
        ip = config["ip"]
        port = config["port"]
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        #   Set timeout
        self.socket.setsockopt(
            zmq.RCVTIMEO, 1000
        )  # Timeout after 1 second (1000 milliseconds)

        #   Define a shared 'stop' flag to control the thread
        self.stop_flag = threading.Event()

        self.last_position = None
        self.last_sent = None
        self._thr = None

    def start(self):
        #   Create and link function to this new thread
        self._thr = threading.Thread(target=self.subscribe_and_process)

        #   Start thread
        self._thr.start()

    def stop(self):
        #   Set the stop flag to signal the thread to exit
        self.stop_flag.set()

        #   Wait for the thread to complete
        self._thr.join()

        #   Close ZMQ socket
        self.socket.close()
        self.context.term()

        #   Confirm with sending a message to the user
        print("Positioner thread successfully terminated.")

    def get_data(self) -> PositionerValue:
        # return last position if its fresh enough or if its changed
        # if self.last_position is None:
        #     return None

        # if self.last_sent is None:
        #     self.last_sent = self.last_position
        #     return self.last_position

        # if self.last_sent is not self.last_position:
        #     self.last_sent = self.last_position
        #     return self.last_position
        return self.last_position

    def subscribe_and_process(self):
        while not self.stop_flag.is_set():
            try:
                # Receive the reply from the server for the first request
                message = self.socket.recv_string()
                self.last_position = json.loads(
                    message, object_hook=PositionerValue.json_decoder
                )
            except zmq.error.Again as e:
                # Handle timeout error
                print("Positioner Thread: Socket receive timed out:", e)
