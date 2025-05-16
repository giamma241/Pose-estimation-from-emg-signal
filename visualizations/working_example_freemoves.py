"""
        _
    .__(.)<  (KWAK)
     \___)
~~~~~~~~~~~~~~~~~~~~
"""

import os
import sys
import time

sys.path.append("../")
from HandPoseStreamer.pose_streamer import Pose_Streamer


def working_example_3():
    # Working example of a TCP stream of pose data from Python to Unity
    # One pose is made of 17 triplets of coordonates, each ending with the ';' delimiter

    # Working example 1 implement a stream of pose data from a csv file at 60Hz

    degrees_of_freedom = 17

    # File paths
    true_csv_path = os.path.join(
        os.path.dirname(__file__), "freemoves\Y_true_freemoves.csv"
    )
    est_csv_path = os.path.join(
        os.path.dirname(__file__), "freemoves\Y_predicted_freemoves.csv"
    )

    # Load and parse true data
    with open(true_csv_path) as f:
        true_lines = [line.strip() for line in f.readlines()]

    true_data = [
        "".join(t + ";" for t in line.split(";")[:degrees_of_freedom])
        for line in true_lines
    ]

    # Load and parse estimated data
    with open(est_csv_path) as f:
        est_lines = [line.strip() for line in f.readlines()]

    est_data = [
        "".join(t + ";" for t in line.split(";")[:degrees_of_freedom])
        for line in est_lines
    ]

    sfreq = 50  # since poses in the data file are recorded at 60Hz
    sending_interval = 1 / sfreq
    ps_true = Pose_Streamer(sfreq, "127.0.0.1", 25001)
    ps_est = Pose_Streamer(sfreq, "127.0.0.1", 25002)
    ps_true.start_streaming()
    ps_est.start_streaming()
    i = 0

    start_time = time.time()
    while True:
        try:
            i %= min(len(true_data), len(est_data))
            ps_true.send_pose(true_data[i])
            ps_est.send_pose(est_data[i])
            i += 1
            time.sleep(
                sending_interval - ((time.time() - start_time) % sending_interval)
            )

        except KeyboardInterrupt:
            ps_true.stop_streaming()
            ps_est.stop_streaming()
            break


if __name__ == "__main__":
    working_example_3()
