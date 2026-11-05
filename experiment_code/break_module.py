'''
This is the break module. It sends triggers, then gives participants a 5 min break.
Resting state recordings are taken during this break.
'''


from psychopy import core, visual, event
import numpy as np
import sounddevice as sd
import argparse
import trigger_module  # trigger_module.send_trigger



def play_beep(duration=2.0, freq=440, fs=44100, volume=0.3):
    """
    Generate a simple sine-wave beep and play it via sounddevice.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = volume * np.sin(2 * np.pi * freq * t)
    sd.play(tone, samplerate=fs)
    sd.wait()


def run_break_routine(
    win=None,
    duration=5.0,
    start_trigger=None,
    end_trigger=None,
    beep_length=1.0,
    break_text_str=(
        "Bitte schließen Sie die Augen und entspannen Sie sich. "
        "Dies ist eine Pause, aber wir zeichnen weiterhin auf, daher bitten wir Sie, "
        "ruhig sitzen zu bleiben. \n\nSie hören einen kurzen Ton, wenn das Experiment fortgesetzt wird"
    ),
    poll_interval=0.1,
):
    """
    Display a break screen, send start/end EEG triggers, then play a beep.
    F10 can be used to skip the break for testing purposes.
    """
    print(
        f"\n--- Starting run_break_routine with target duration: {duration} min "
        f"({duration*60:.2f}s) ---"
    )
    function_overall_timer = core.Clock()

    total_seconds = duration * 60.0
    wait_before_beep = max(total_seconds - beep_length, 0.0)
    skip_break = False
    created_our_own_win = False

    if win is None:
        print("  No window provided, creating a new one for the break routine.")
        win = visual.Window(
            fullscr=True,
            units="height",
            color="gray",
            colorSpace="rgb",
            useRetina=True,
        )
        created_our_own_win = True

    win.color = "gray"
    win.colorSpace = "rgb"

    # Prepare text stim
    break_text = visual.TextStim(
        win=win,
        text=break_text_str,
        color="white",
        height=0.05,
    )
    break_text.draw()
    win.flip()
    event.clearEvents()

    if start_trigger is not None:
        trigger_module.send_trigger(int(start_trigger))
        print(f"  Sent start_trigger: {start_trigger}")

    print(f"  Targeting wait_before_beep: {wait_before_beep:.3f}s")
    pre_beep_loop_timer = core.Clock()

    if wait_before_beep > 0:
        timer = core.Clock()
        while timer.getTime() < wait_before_beep:
            # --- MODIFIED: Changed key checking logic ---
            keys = event.getKeys(keyList=["f10", "escape"])
            if 'f10' in keys:
                skip_break = True
                print("  Break skipped by user with F10 key.")
                break # Exit the wait loop to skip the break
            if 'escape' in keys:
                print("  Escape key pressed during break. Exiting experiment.")
                core.quit() # Exit the entire experiment
            
            # --- MODIFIED: Draw the skip prompt inside the loop too ---
            break_text.draw()
            win.flip()
            # We remove core.wait() and just let the loop run, as win.flip() syncs to the screen refresh rate
    
    actual_pre_beep_wait_duration = (
        0.0 if wait_before_beep <= 0 else pre_beep_loop_timer.getTime()
    )
    print(f"  Actual pre-beep wait duration: {actual_pre_beep_wait_duration:.3f}s")

    if end_trigger is not None:
        trigger_module.send_trigger(int(end_trigger))
        print(f"  Sent end_trigger: {end_trigger} ")

    actual_beep_play_duration = 0.0
    if not skip_break:
        print(f"  Playing beep (target {beep_length:.3f}s)...")
        beep_play_timer = core.Clock()
        play_beep(duration=beep_length, freq=440)
        actual_beep_play_duration = beep_play_timer.getTime()
        print(f"  Actual beep play duration: {actual_beep_play_duration:.3f}s")
    else:
        print("  Beep skipped due to user abort.")

    win.flip()
    if created_our_own_win:
        print("  Closing self-created window...")
        win.close()

    total_function_time = function_overall_timer.getTime()
    print(f"  Total run_break_routine function time: {total_function_time:.3f}s")
    print(f"--- Finished run_break_routine ---\n")


# --- Standalone execution block (No changes needed here) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a standalone break routine for a PsychoPy experiment."
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=5.0,
        help="Duration of the break in minutes. Default: 5.0",
    )
    parser.add_argument(
        "--start-trigger",
        type=int,
        default=None,
        help="Integer trigger code to send at break start. Default: None.",
    )
    parser.add_argument(
        "--end-trigger",
        type=int,
        default=None,
        help="Integer trigger code to send at break end / beep onset. Default: None.",
    )
    parser.add_argument(
        "--beep-length",
        type=float,
        default=1.0,
        help="Length (s) of the beep played at break end. Default: 1.0.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Polling interval (s) for checking ESC during break. Default: 0.1.",
    )
    args = parser.parse_args()

    run_break_routine(
        duration=args.duration,
        start_trigger=args.start_trigger,
        end_trigger=args.end_trigger,
        beep_length=args.beep_length,
        poll_interval=args.poll_interval,
    )

    core.quit()