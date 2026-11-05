#!/usr/bin/env python

# -*- coding: utf-8 -*-


import os
import random
import numpy as np
import pandas as pd
from psychopy import visual, core, event, data, gui
from glob import glob
import imageio
from trigger_module import send_trigger

def load_process_images(img_dir):

    """
    Load and process images from the specified directory.
    The images are normalized to [-1, 1] and masked with a circular mask.
    """
    images = []
    # Use a non-recursive glob; adjust to recursive if needed (see "**" glob argument)

    for f in glob(os.path.join(img_dir, '*.jpg')):
        img = imageio.imread(f)
        img = (img / 255.0) * 2 - 1  # Normalize pixel values
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(center_x, center_y)
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[~mask] = 0
        images.append(np.flipud(img))  # Flip vertically for PsychoPy compatibility
    return images

# In gradCPT_module.py, near the top:
# ... (imports like os, random, np, pd, visual, core, event, data, gui, glob, imageio) ...
# ------------------------------------------------------------------
# STIMULI DIRECTORIES AND PRELOADED STIMULI OBJECTS
# ------------------------------------------------------------------


MOUNTAIN_DIR = os.path.join("gradcpt", "scenes5", "mountain")

CITY_DIR = os.path.join("gradcpt", "scenes5", "city")


# Preload the images from disk into global variables

MOUNTAIN_IMAGES = load_process_images(MOUNTAIN_DIR) # Or pass abs_mountain_dir

CITY_IMAGES = load_process_images(CITY_DIR)     # Or pass abs_city_dir


# Add a critical check here:
if not MOUNTAIN_IMAGES:
    # Using the absolute path in the error message for clarity
    raise FileNotFoundError(f"CRITICAL ERROR: No mountain images loaded from '{abs_mountain_dir}'. "
                          f"Please check that the 'scenes5/mountain' directory exists relative to your script "
                          f"and contains '*.jpg' files.")
if not CITY_IMAGES:
    raise FileNotFoundError(f"CRITICAL ERROR: No city images loaded from '{abs_city_dir}'. "
                          f"Please check that the 'scenes5/city' directory exists relative to your script "
                          f"and contains '*.jpg' files.")


def get_go_key():
    """
    Returns the key for the 'Go' response.
    """
    return 'space'  # Use the space bar as the single Go key


def init_window(fullscr=True, win_size=(1920, 1080)):
    """
    Create a PsychoPy Window.
    """

    return visual.Window(
        size=win_size,
        fullscr=fullscr,
        color=[grey],
        units='height',
        allowGUI=False
    )

def show_text(win, text_str, keys=['space'], wrap_width=1.5, height=0.05):
    """
    Draw a block of text and wait for one of `keys`.
    """
    stim = visual.TextStim(win, text=text_str, height=height, wrapWidth=wrap_width)
    stim.draw()
    win.flip()
    event.waitKeys(keyList=keys)


import random


def create_trial_sequence(n_trials, go_prob):
    """
    Generates a pseudo-random trial sequence with two key constraints:
    1. The overall probability of 'go' trials is met as closely as possible.
    2. No two 'nogo' trials can appear consecutively.

    Args:
        n_trials (int): Total number of trials in the sequence.
        go_prob (float): The desired proportion (0.0 to 1.0) of 'go' trials.

    Returns:
        list: A shuffled list of 'go' and 'nogo' strings meeting the constraints.
    """
    # 1. Calculate the exact number of go and nogo trials
    n_go = round(n_trials * go_prob)
    n_nogo = n_trials - n_go

    # 2. Check if the sequence is possible. We need at least n_nogo - 1 'go'
    #    trials to act as separators between the 'nogo' trials.
    if n_go < n_nogo - 1:
        raise ValueError(
            f"Cannot create a sequence with no consecutive 'nogo' trials. "
            f"Need at least {n_nogo - 1} 'go' trials, but only {n_go} are available."
        )

    # 3. Create 'nogo' trials, each followed by a mandatory 'go' trial separator
    #    This is the core logic that prevents consecutive 'nogo's.
    sequence_chunks = [['nogo', 'go'] for _ in range(n_nogo)]

    # 4. Add the remaining 'go' trials to the list of items to be shuffled
    n_go_remaining = n_go - n_nogo
    sequence_chunks.extend(['go'] * n_go_remaining)

    # 5. Shuffle the chunks. Each element is either a ['nogo', 'go'] pair or a single 'go'.
    random.shuffle(sequence_chunks)

    # 6. Flatten the shuffled list of chunks into the final sequence
    final_sequence = []
    for item in sequence_chunks:
        if isinstance(item, list):
            final_sequence.extend(item)
        else:
            final_sequence.append(item)
            
    # As a final check, trim any excess trials if rounding created them
    return final_sequence[:n_trials]




def run_trial(
    win,
    img_previous,         # The image array to fade OUT from (can be None)
    img_current,          # The image array to fade IN to
    cond,                 # "go" for City(Go), "nogo" for Mountain(NoGo)
    go_key,               # The key for a Go response (e.g., 'space')
    transition_time,      # The duration of the cross-fade
):
    """
    Presents a smooth cross-fade transition from a previous to a current image,
    collects a single 'go' response, and sends triggers.

    Triggers (in-block, 4-bit):
      1 = Stimulus: City (Go)
      2 = Stimulus: Mountain (NoGo)
      3 = Response: Go key pressed
      4 = response missed
      5 = Responded go on no-go trial
    """
    from functools import partial

    # ---- Map to 4-bit stim codes ----
    stim_code = 1 if cond == "go" else 2

    # ---- Prepare stimuli for the transition ----
    # The new, incoming image starts completely transparent
    stim_incoming = visual.ImageStim(win, image=img_current, size=(0.5, 0.5), opacity=0.0)

    # The old, outgoing image (if one exists) starts fully opaque
    stim_outgoing = None
    if img_previous is not None:
        stim_outgoing = visual.ImageStim(win, image=img_previous, size=(0.5, 0.5), opacity=1.0)

    # ---- Prep timing, input, and the initial trigger ----
    resp_clock = core.Clock()
    transition_clock = core.Clock()
    event.clearEvents(eventType='keyboard')

    # Reset the response clock and fire the stimulus trigger on the very first frame of the animation
    win.callOnFlip(resp_clock.reset)
    win.callOnFlip(partial(send_trigger, stim_code))

    # ---- Main Animation & Response Loop ----
    # This loop runs on every frame for the duration of the transition
    response_key_name = None
    rt = None
    go_key_l = (go_key or "").lower()

    while transition_clock.getTime() < transition_time:
        # 1. Calculate the current fade progress (alpha)
        elapsed = transition_clock.getTime()
        alpha = min(elapsed / transition_time, 1.0) # Clamp to 1.0

        # 2. Update stimulus opacities based on progress
        stim_incoming.opacity = alpha
        if stim_outgoing:
            stim_outgoing.opacity = 1.0 - alpha

        # 3. Draw the stimuli (outgoing first, then incoming on top)
        if stim_outgoing:
            stim_outgoing.draw()
        stim_incoming.draw()

        # 4. Flip the window to show the updated frame
        win.flip() # The callOnFlip trigger fires on the first pass here

        # 5. Poll for responses (without interrupting the animation)
        keys = event.getKeys(keyList=[go_key, 'escape'], timeStamped=resp_clock)
        if keys and response_key_name is None: # Process only the first valid keypress
            key, t_stamp = keys[0]
            k = key.lower()

            if k == 'escape':
                win.close()
                core.quit()
                return 'escape', 0.0

            if k == go_key_l:
                response_key_name = k
                rt = t_stamp
                send_trigger(3) # Fire response trigger

    # The loop has finished, but we still need to collect responses for the full SOA
    # This part of the code is not strictly necessary if SOA == transition_time
    # but it ensures responses right at the end are still caught.
    while resp_clock.getTime() < transition_time:
        keys = event.getKeys(keyList=[go_key, 'escape'], timeStamped=resp_clock)
        if keys and response_key_name is None: # Process only the first valid keypress
            key, t_stamp = keys[0]
            k = key.lower()
            if k == 'escape':
                win.close()
                core.quit()
                return 'escape', 0.0
            if k == go_key_l:
                response_key_name = k
                rt = t_stamp
                send_trigger(3)

    # ---- [NEW] Final trigger logic based on outcome ----
    # Check for missed Go trials or incorrect NoGo responses
    if cond == "go" and response_key_name is None:
        send_trigger(4)  # Trigger for a missed response on a Go trial
    elif cond == "nogo" and response_key_name is not None:
        send_trigger(5)  # Trigger for an incorrect response on a NoGo trial

    return response_key_name, rt





def run_block(win,
              block_num,
              total_blocks,
              n_trials,
              go_prob,
              cities,
              mountains,
              go_key,
              transition_time,
              save_fn,
              max_run_go):
              
    # Note the updated function call to our new sequence generator
    sequence = create_trial_sequence(n_trials=n_trials, go_prob=go_prob)
    
    show_text(
        win,
        f"Block {block_num} of {total_blocks}\nPress SPACE to begin"
    )

    # --- KEY CHANGE: We will now select images one-by-one inside the loop ---
    previous_image = None
    last_city_choice = None
    last_mountain_choice = None
    
    for t, cond in enumerate(sequence, start=1):
        # 1. Select the image for the current trial, ensuring no immediate repeats
        current_image = None
        if cond == 'go':
            # Create a list of candidate images, excluding the last one shown
            candidates = [img for img in cities if img is not last_city_choice]
            if not candidates: candidates = cities # Fallback if only one image exists
            current_image = random.choice(candidates)
            last_city_choice = current_image
        else: # condition is 'nogo'
            candidates = [img for img in mountains if img is not last_mountain_choice]
            if not candidates: candidates = mountains # Fallback
            current_image = random.choice(candidates)
            last_mountain_choice = current_image

        # 2. Run the trial with the transition
        resp, rt = run_trial(
            win,
            img_previous=previous_image,
            img_current=current_image,
            cond=cond,
            go_key=go_key,
            transition_time=transition_time
        )
        
        # 3. Save data
        save_fn(block_num, t, cond, resp, rt)
        
        # 4. After the trial, the current image becomes the next trial's previous image
        previous_image = current_image

    # After the block, draw the final image statically
    if previous_image is not None:
        final_stim = visual.ImageStim(win, image=previous_image, size=(0.5, 0.5))
        final_stim.draw()
        win.flip()
        core.wait(0.5)

def make_save_fn(participant, exp_name, go_key): # <-- 'go_key' must be here
    """
    Return a save_trial_data function that writes rows to a CSV.
    """
    data_dir = os.path.join("data", "gradcpt")
    os.makedirs(data_dir, exist_ok=True)

    def save_trial_data(block, trial, condition, response_key_name, rt):
        fname = os.path.join(
            data_dir,
            f"{participant}gradCPT_block-{block:02}.csv"
        )

        # --- Go/No-Go Correctness Logic ---
        is_correct_response = False
        if condition == 'go':
            # Correct if they pressed the go key
            is_correct_response = (response_key_name == go_key)
        elif condition == 'nogo':
            # Correct if they did NOT press any key
            is_correct_response = (response_key_name is None)
        
        correct_val = int(is_correct_response)

        row = {
            'participant': participant,
            'block': block,
            'trial': trial,
            'condition': condition,
            'stimulus_type': 'City' if condition == 'go' else 'Mountain',
            'response': response_key_name,
            'rt': rt if rt is not None else -1,
            'correct': correct_val,
            'go_key_assigned': go_key
        }
        df = pd.DataFrame([row])
        if not os.path.exists(fname):
            df.to_csv(fname, index=False)
        else:
            df.to_csv(fname, mode='a', header=False, index=False)
            
    return save_trial_data


# ------------------------------------------------------------------

# MAIN EXPERIMENT FUNCTION

# ------------------------------------------------------------------


def run_gradcpt(
        win,
        participant,
        session,
        n_trials: int = 200, # default n_trials
        n_blocks: int = 1, #default repetitions
        go_prob: float = 0.90, # Likelihood of an image being city
        transition_time: float = 0.7, # Default time from 0-1 for the images to transition.
        max_run_go: int = 3
):
    """
    Run a Go/No-Go GradCPT experiment block-by-block.
    """
    exp_name = "gradcpt_gonogo"
    exp_info = {
        "participant": participant,
        "session": session,
        "date": data.getDateStr(),
        "go_prob": go_prob,
        "n_blocks": n_blocks,
        "n_trials": n_trials,
    }

    cities = CITY_IMAGES
    mountains = MOUNTAIN_IMAGES
    go_key = get_go_key() # Use the new function

    # Update the call to the data saving function
    save_fn = make_save_fn(
        participant=participant,
        exp_name=exp_name,
        go_key=go_key
    )

    # --- CRITICAL: Update instructions for Go/No-Go ---
    instr_txt = (
        f"In dieser Aufgabe:\n\n"
        f"- Drücken Sie die LEERTASTE, wenn Sie eine STADTszene sehen.\n\n"
        f"- Drücken Sie KEINE Taste, wenn Sie eine BERGszene sehen.\n\n"
        "Reagieren Sie so schnell und genau wie möglich.\n\n"
        "Drücken Sie die LEERTASTE, um zu beginnen."
    )
    show_text(win, instr_txt)

    for blk in range(1, n_blocks + 1):
        # Update the call to run_block with the new parameters
        run_block(
            win=win,
            block_num=blk,
            total_blocks=n_blocks,
            n_trials=n_trials,
            go_prob=go_prob,
            cities=cities,
            mountains=mountains,
            go_key=go_key,
            transition_time=transition_time,
            save_fn=save_fn,
            max_run_go=max_run_go
        )


def run_gradCPT_task(win, io, participant_id, **kwargs):
    """
    Adapter function to run the GradCPT task from the main experiment controller.

    It accepts the standard arguments from the controller (`win`, `io`, `participant_id`)
    and calls the module's internal `run_gradcpt` function.

    Args:
        win: The shared PsychoPy window object.
        io: The shared PsychoPy ioHub object (ignored by this task).
        participant_id (str): The participant ID from the main script.
        **kwargs: Catches any other parameters passed from the task list.
    """
    print(f"--- Launching PsychoPy GradCPT Task for sub-{participant_id} ---")

    # Call the core experiment function, mapping the parameters.
    # We use a default 'session' value as it's not in the main script's context.
    run_gradcpt(
        win=win,
        participant=participant_id,
        session='01',  # Default session number
        **kwargs       # Pass along any extra params like n_trials, etc.
    )

    print("--- PsychoPy GradCPT Task complete. Returning control. ---")


# --- Standalone Execution Logic for Testing ---

if __name__ == '__main__':
    # This block allows you to test the module by running it directly
    win = None
    try:
        # Get participant info with a GUI
        exp_info = {'participant': 'GRADCPT_TEST', 'session': '01'}
        dlg = gui.DlgFromDict(dictionary=exp_info, title='GradCPT')
        if not dlg.OK:
            core.quit()

        # Create a window just for this test run
        win = visual.Window(size=(1280, 800), fullscr=False, color='gray', units='height')

        # Run the task using the adapter function (good for testing the integration point)
        run_gradCPT_task(
            win=win,
            io=None, # Not used
            participant_id=exp_info['participant'],
            n_trials=50, # Override default for quick testing
            n_blocks=1
        )

        # Final message
        show_text(win, "Test complete. Thank you!", height=0.07)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the window is closed and PsychoPy quits properly
        if win:
            win.close()
        core.quit()