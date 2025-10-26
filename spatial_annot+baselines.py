import hydra
from omegaconf import DictConfig
import cv2
import matplotlib.pyplot as plt
import os
import json

# Make sure to enable X11 forwarding for this!

def annotate_frame(frame, prompt):
    '''
    Takes a frame and prompt, visualizes them together, and allows for user annotation with a click on the correct pixel.
    '''
    # Display the frame, the prompt should just be the caption of the image
    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.set_title(prompt)

    # Allow the user to click on the correct pixel and return the coordinates
    coords = []
    def onclick(event):
        ix, iy = int(event.xdata), int(event.ydata)
        print (f'x = {ix}, y = {iy}')
        coords.append((ix, iy))

        # Remove any previous red dots, but keep the title and the image
        ax.clear()
        ax.set_title(prompt)
        ax.imshow(frame)

        # Show a red dot there
        ax.scatter(ix, iy, color='red', marker='o', s=100)
        plt.draw()

    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    print(f"Coords: {coords}")

    # Returning only the last click
    return coords[-1]

def gpt5_baseline(frame, prompt):
    raise NotImplementedError("Not implemented yet")


def create_baselines(frame, prompt):
    raise NotImplementedError("Not implemented yet")


    
@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # Get the preprocessed root
    preprocessed_root = cfg.preprocessed_root

    # Read the config all the way to the clips
    clips_cfg = cfg.clips

    for clip in clips_cfg:
        # Read the clip config
        clip_cfg = clip
        print(f"Clip config: {clip_cfg}")

        # Get spatial eval file
        spatial_eval_file = clip_cfg.spatial_eval_file

        # TODO: should also implement that if there is already an annotation key, we do not overwrite it and skip that

        with open(spatial_eval_file, 'r') as f:
            spatial_eval_data = json.load(f)
        spatial_prompts = spatial_eval_data['spatial_prompts']
        spatial_prompts_frames = spatial_eval_data['spatial_prompts_frames']

        # Make sure there are as many prompts as frames
        assert len(spatial_prompts) == len(spatial_prompts_frames), "There must be as many prompts as frames"

        # Relevant to obtain the video frames
        video_dir = clip_cfg.name
        
        annotations = []
        # TODO: not implemented yet
        # baselines = []

        # Pair them, iterate over the pairs
        for prompt, frame in zip(spatial_prompts, spatial_prompts_frames):
            # Read the frame with cv2
            frame_path = os.path.join(preprocessed_root, video_dir, "images", f"{frame}")
            print(f"Frame path: {frame_path}")
            # Convert from bgr to rgb
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create ground truth annotation
            annotation = annotate_frame(frame, prompt)
            annotations.append(annotation)

            # # Create baselines
            # baselines = create_baselines(frame, prompt)
            # baselines.append(baselines)

        print(f"Annotations: {annotations}")
        
        # Store in the json spatial eval file again; don't overwrite the file, just add to it
        spatial_eval_data['annotations'] = annotations
        with open(spatial_eval_file, 'w') as f:
            json.dump(spatial_eval_data, f, indent=4)


if __name__ == "__main__":
    main()