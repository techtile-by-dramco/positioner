import zmq
import threading
import json
import numpy as np
import numpy as np
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import samplerate
import asyncio
import qtm

class PositionerValues(object):
    def __init__(self, arr=None):
        self.values = arr

    def load_file(self, file:str):
        self.values = np.load(file, allow_pickle=True)

    @staticmethod
    def from_xyz(x,y,z):
        # todo allow to include time and rotation matrix

        arr = np.asarray([PositionerValue(-1.0, _x, _y, _z, None) for _x, _y, _z in zip(x,y,z)])
        return PositionerValues(arr)

    def get_x_positions(self):
        return np.asarray([pos.x for pos in self.values])

    def get_y_positions(self):
        return np.asarray([pos.y for pos in self.values])

    def get_z_positions(self):
        return np.asarray([pos.z for pos in self.values])

    def reduce_to_grid_size(self, size=0.1) -> PositionerValues:
        x_rounded = np.round(self.get_x_positions() / size) * size
        y_rounded = np.round(self.get_y_positions() / size) * size
        z_rounded = np.round(self.get_z_positions() / size) * size

        # todo allow to include time and rotation matrix

        return PositionerValues.from_xyz(x_rounded, y_rounded, z_rounded)


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
    # todo allow to specify backend (ZMQ or direct)
    def __init__(self, config: dict, backend="direct") -> None:
        # TODO backend specify in config or as extra param?
        # TODO replace backend str by enum

        self.ip = config["ip"]
        self.port = config["port"]
        self.backend = backend
        self.wanted_body = config["wanted_body"]
        if backend == "zmq":
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(f"tcp://{self.ip}:{self.port}")
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

    @staticmethod
    def create_body_index(xml_string):
        """Extract a name to index dictionary from 6dof settings xml"""
        xml = ET.fromstring(xml_string)

        body_to_index = {}
        for index, body in enumerate(xml.findall("*/Body/Name")):
            body_to_index[body.text.strip()] = index

        return body_to_index

    @staticmethod
    def check_NaN(position, rotation):
        return np.isnan(float(position[0]))

    # TODO check if possible in class
    async def main_async(self, wanted_body, measuring_time):
        # Connect to qtm
        connection = await qtm.connect(self.ip) 

        # Connection failed?
        if connection is None:
            print("Qualisys: Failed to connect")
            return

        # Take control of qtm, context manager will automatically release control after scope end
        async with qtm.TakeControl(connection, "PW"):  #ENTER PW
            await connection.new()

        # Get 6dof settings from qtm
        xml_string = await connection.get_parameters(parameters=["6d"])
        body_index = PositionerClient.create_body_index(xml_string)

        temp_data = []
        def on_packet(packet):
            info, bodies = packet.get_6d()
            framenumber = packet.framenumber  # number of the frame/position estimate
            body_count = info.body_count  # amount of tracked bodies
            now = datetime.now()

            if wanted_body is not None and wanted_body in body_index:
                # Extract one specific body
                wanted_index = body_index[wanted_body]
                position, rotation = bodies[wanted_index]

                if not PositionerClient.check_NaN(position, rotation):
                    data = dict(t=now.strftime("%H:%M:%S"),
                                x=position[0] / 1000,  # x-position in [m]
                                y=position[1] / 1000,  # y-position in [m]
                                z=position[2] / 1000,  # z-position in [m]
                                rotation_matrix=rotation[0])

                    temp_data.append(data)
                    np.save('temp_data_qualisys\\temp_data_qualisys.npy', temp_data)

                else:
                    print('Qualisys: No object detected')
            else:
                # Print all bodies
                print('Qualisys: NO BODY FOUND')

        # Start streaming frames
        await connection.stream_frames(components=["6d"], on_packet=on_packet)

        # Wait asynchronously some time
        await asyncio.sleep(measuring_time)

        # Stop streaming
        await connection.stream_frames_stop()

    def get_Qualisys_Position(self, wanted_body, measuring_time):
        asyncio.get_event_loop().run_until_complete(self.main_async(wanted_body, measuring_time))

    @staticmethod
    def average_qualisys_data():
        data_list = np.load("temp_data_qualisys\\temp_data_qualisys.npy")
        n = len(data_list)

        # Sum all values using list comprehensions and NumPy for the rotation matrix
        x_avg = sum(data['x'] for data in data_list) / n
        y_avg = sum(data['y'] for data in data_list) / n
        z_avg = sum(data['z'] for data in data_list) / n

        # Convert time to timedelta, then sum them up and average
        time_sum = sum((
            timedelta(hours=int(data['t'][:2]), minutes=int(data['t'][3:5]), seconds=int(data['t'][6:]))
            for data in data_list
        ), timedelta(0))  # Start with timedelta(0) instead of an integer

        avg_time = time_sum / n  # Average timedelta

        # Convert average timedelta back to HH:MM:SS format
        avg_time_str = str(avg_time).split(".", maxsplit=1)[0]  # Remove microseconds

        # Sum and average rotation matrices
        rotation_matrix_avg = sum(np.array(data['rotation_matrix']) for data in data_list) / n

        return {
            't': avg_time_str,
            'x': x_avg,
            'y': y_avg,
            'z': z_avg,
            'rotation_matrix': rotation_matrix_avg.tolist()  # Convert back to list
        }

    def start(self):

        if self.backend == "zmq":
            #   Create and link function to this new thread
            self._thr = threading.Thread(target=self.subscribe_and_process)

            #   Start thread
            self._thr.start()

    def stop(self):
        if self.backend == "zmq":
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

        if self.backend == "direct":
            self.get_Qualisys_Position(self.wanted_body, 0.03)
            # Average positioning data recorded with Qualisys in given timeframe
            self.last_position = PositionerClient.average_qualisys_data()

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
