import cv2
import os
import argparse
import glob

import yaml


def frames_to_video(
    start_index,
    end_index,
    output_dir=".",
    output_filename="output_video.mp4",
    frame_rate=30,
    frame_folder=".",
):
    os.makedirs(output_dir, exist_ok=True)
    # Initialize variables
    images = []
    # Loop through each frame index within the specified range
    for index in range(start_index, end_index + 1):
        # Create the filename for the current index
        filename = f"color_frame_{index:04d}.png"
        # Construct the full file path
        filepath = os.path.join(frame_folder, filename)
        # Check if the file exists
        if os.path.isfile(filepath):
            # Read the image using OpenCV
            img = cv2.imread(filepath)
            if img is not None:
                # Append the image to the list if it's successfully loaded
                images.append(img)
            else:
                print(f"Warning: File {filename} not successfully loaded.")
        else:
            print(f"Warning: File {filename} does not exist.")
    # Check if any images have been loaded
    if not images:
        print("No images found, video creation aborted.")
        return
    # Get the width, height from the first image (assuming all images have the same dimensions)
    height, width, _ = images[0].shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Codec for mp4 output
    output_path = os.path.join(output_dir, output_filename)
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Write each image to the video file
    for image in images:
        video_writer.write(image)

    # Release the video writer
    video_writer.release()

    print(f"Video saved as {output_path}")


# Usage example:
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--frame-folder",
    default="/home/ros/data/coffee_room_events/batch_1/images/color/",
)
parser.add_argument("-s", "--start-index", type=int, default=1)
parser.add_argument("-e", "--end-index", type=int, default=10)
parser.add_argument("-r", "--frame-rate", type=int, default=5)
parser.add_argument(
    "-d",
    "--output-dir",
    type=str,
    default="/home/ros/data/coffee_room_events/videos",
)
parser.add_argument(
    "-o",
    "--output-filename",
    type=str,
    default="output.mp4",
)
args = parser.parse_args()
# output_filename = "output_video.mp4"

event_dirs = [
    "/home/ros/data/coffee_room_events/batch_1/events_gt/",
    "/home/ros/data/coffee_room_events/batch_2/events_gt/",
    "/home/ros/data/coffee_room_events/batch_3/events_gt/",
    "/home/ros/data/coffee_room_events/batch_4/events_gt/",
]

yaml_files = []
cam_config_file = "../configs/camera/astra2.yaml"
for directory in event_dirs:
    yaml_files += glob.glob(os.path.join(directory, "*.yaml")) + glob.glob(
        os.path.join(directory, "*.yml")
    )

idx = 0
for event_param_file in sorted(yaml_files):
    idx += 1
    with open(event_param_file, "r") as f:
        event_data = yaml.safe_load(f)
    event_raw_data_path = event_data.get("image_path")
    event_dir = os.path.dirname(os.path.abspath(event_param_file))
    frame_folder = os.path.join(event_dir, event_raw_data_path) + "/color"
    from_frame = event_data.get("from_frame")
    to_frame = event_data.get("to_frame")

    output_filename = f"clip_{str(idx).zfill(2)}.mp4"
    frames_to_video(
        from_frame,
        to_frame,
        output_dir=args.output_dir,
        output_filename=output_filename,
        frame_rate=args.frame_rate,
        frame_folder=frame_folder,
    )
