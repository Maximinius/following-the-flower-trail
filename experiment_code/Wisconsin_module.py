#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 12:50:23 2025

@author: max
"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from trigger_module import send_trigger # cursom trigger script - handles timings, just needs to be told which trigger value to send through the parallel port
import itertools
import os
import random
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import argparse
import pygame


# ── WCST ENVIRONMENT PREP ─────────────────────────────────────────────────────
def wcst_prepare_environment():
    ctx = {'cwd_before': os.getcwd(), 'io_server_preexisting': False}

    try:
        event.clearEvents('keyboard')
        event.clearEvents('mouse')
    except Exception:
        pass

    io_server = getattr(deviceManager, 'ioServer', None)
    if io_server is not None:
        ctx['io_server_preexisting'] = True
        try:
            for dev in io_server.devices.values():
                try:
                    dev.clearEvents()
                except Exception:
                    pass
        except Exception:
            pass

    # clear any prior wcstKeyboard adapter (fresh each run)
    if hasattr(deviceManager, 'wcstKeyboard'):
        delattr(deviceManager, 'wcstKeyboard')

    return ctx


# ------------------------------------------------------------------
# ioHub keyboard adapter: provides .getKeys(keyList=None) interface
# compatible with the bits of WCST that only need Escape detection.
# ------------------------------------------------------------------
def _make_iohub_keyboard_adapter(io_server):
    """
    Return a lightweight wrapper around the ioHub Keyboard device.

    If no ioHub server or device is available, returns a fallback that
    delegates to event.getKeys().
    """
    try:
        kb_dev = io_server.getDevice('keyboard')
    except Exception:
        kb_dev = None

    if kb_dev is None:
        # fallback wrapper
        class _EventKB:
            def getKeys(self, keyList=None):
                return event.getKeys(keyList=keyList)
            def clearEvents(self):
                try:
                    event.clearEvents('keyboard')
                except Exception:
                    pass
        return _EventKB()

    # ioHub keyboard -> adapter
    class _IoHubKB:
        def getKeys(self, keyList=None):
            # ioHub key events: kb_dev.getEvents() returns a list of events;
            # we filter by keyList if provided and return a list of .key value strings.
            evts = kb_dev.getEvents()
            if not evts:
                return []
            keys = [e.key for e in evts if (keyList is None or e.key in keyList)]
            return keys
        def clearEvents(self):
            try:
                kb_dev.clearEvents()
            except Exception:
                pass
    return _IoHubKB()


# ── BULLETPROOF DEVICE SETUP ──────────────────────────────────────────────────
def setupDevices(expInfo, thisExp, win):
    """
    WCST device init that *always* provides an ioHub keyboard adapter,
    regardless of any preexisting PTB keyboard used by earlier tasks.
    Never raises.
    """
    io_server = getattr(deviceManager, 'ioServer', None)

    if io_server is None:
        # Launch ioHub. If it fails, we'll still make a fallback adapter.
        try:
            io_server = io.launchHubServer(
                window=win,
                Keyboard=dict(use_keymap='psychopy'),
                Experiment=dict(filename=thisExp.dataFileName)
            )
            deviceManager.ioServer = io_server
        except Exception:
            io_server = None
            deviceManager.ioServer = None

    # Build adapter (ioHub if possible; else event fallback)
    kb_adapter = _make_iohub_keyboard_adapter(io_server)

    # Stash adapter in deviceManager under predictable attr name
    deviceManager.wcstKeyboard = kb_adapter

    return True




import numpy as np 
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import randint, normal, shuffle, choice as randchoice
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard


def create_trial_card_stim(win, 
                           initial_image: str = 'default.jpg', 
                           pos: tuple = (0, -0.2), 
                           size: tuple = (0.25, 0.25)
                          ) -> visual.ImageStim:
    """
    Create and return a PsychoPy ImageStim for the 'trial card' placeholder.
    
    Parameters
    ----------
    win : psychopy.visual.Window
        The window in which the stimulus will be drawn.
    initial_image : str
        Path (relative or absolute) to the placeholder image. 
        You can overwrite this later with .setImage(...) on each trial.
    pos : (x, y)
        The (x, y) position (in window units) where the card should appear.
        Default is (0, -0.2) (centered horizontally, slightly below center vertically).
    size : (width, height)
        The size of the card (as a fraction of the window height, if your window uses 'height'-units).
    
    Returns
    -------
    trial_card_stim : psychopy.visual.ImageStim
        An ImageStim that you can call .setImage(...) on each trial.
    """
    trial_card = visual.ImageStim(
        win=win,
        name='trial_card',
        image=initial_image,
        pos=pos,
        size=size,
        ori=0.0,
        color=[1, 1, 1],
        colorSpace='rgb',
        interpolate=True,
    )
    return trial_card

# ─── Mouse helpers ────────────────────────────────────────────────────────────
def check_for_click(mouse, clickable, trigger_map=None):
    """
    Return .name of the first clickable object under mouse
    that has just been clicked.  None → no fresh click.

    Automatically debounces by remembering button state for each mouse.
    Optionally sends a trigger defined in trigger_map.
    """
    if not hasattr(check_for_click, "_prev_button_state"): # Renamed for clarity
        check_for_click._prev_button_state = {} # Stores state per mouse object
    
    # Use a unique ID for the mouse object if available, otherwise the object itself
    mouse_id = id(mouse) 
    prev_buttons = check_for_click._prev_button_state.get(mouse_id, (0, 0, 0))

    current_buttons = mouse.getPressed()    # current button state
    clicked_name = None
    
    # Check if button state *changed* AND at least one button is currently pressed
    if current_buttons != prev_buttons and sum(current_buttons) > 0:
        # Check only for the button that went from up to down (a new press)
        # This simple check is fine if any button press is a valid click start
        for obj in clickable:
            if obj.contains(mouse):
                clicked_name = obj.name
                if trigger_map and clicked_name in trigger_map:
                    send_trigger(trigger_map[clicked_name])
                break 
    check_for_click._prev_button_state[mouse_id] = current_buttons # Update stored state
    return clicked_name


# --- Setup global variables (available in all functions) ---
deviceManager = hardware.DeviceManager()
_thisDir = os.path.dirname(os.path.abspath(__file__))
psychopyVersion = '2025.1.0'
expName = 'Wisconsin Card Sorting Task' 
expVersion = ''
runAtExit = []
expInfo = {
    'participant': '', 'session': '001',
    'date|hid': data.getDateStr(), 'expName|hid': expName,
    'expVersion|hid': expVersion, 'psychopyVersion|hid': psychopyVersion,
}

PILOTING = core.setPilotModeFromArgs()
_fullScr = True
_winSize = [1920, 1080]
if PILOTING:
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        _winSize = prefs.piloting['forcedWindowSize']
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

# --- Function definitions from original script (showExpInfoDlg, setupData, etc.) ---
# These functions (showExpInfoDlg, setupData, setupLogging, setupWindow, setupDevices, pauseExperiment) 
# are assumed to be correct and are kept as they are. For brevity, I'm not repeating them here,
# but they should be in your final script.

def showExpInfoDlg(expInfo):
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True)
    if not dlg.OK: core.quit()
    return expInfo

def setupData(expInfo, dataDir=None):
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    if dataDir is None:
        dataDir = _thisDir
    
    filename = u'data/wisconsin/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion, extraInfo=expInfo, runtimeInfo=None,
        originPath=os.path.join(_thisDir, 'Wisconsin Card Sorting Task.py'), # Adjusted path
        savePickle=True, saveWideText=True, dataFileName=os.path.join(dataDir, filename),
        sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    return thisExp

def configure_logging_for_trials(level: str = 'exp'):
    """
    Ensure the PsychoPy logger is verbose enough for per-trial entries.

    level: one of {'debug','info','data','exp','warning','error','critical'}
    """
    try:
        logging.console.setLevel(level)
    except Exception:
        pass
    try:
        # Update all existing LogFile handlers too (if any)
        for h in logging.root.handlers:
            if isinstance(h, logging.LogFile):
                h.setLevel(logging.getLevel(level))
    except Exception:
        pass


def setupLogging(filename):
    if PILOTING: logging.console.setLevel(prefs.piloting['pilotConsoleLoggingLevel'])
    else: logging.console.setLevel('warning')
    logFile = logging.LogFile(filename+'.log')
    if PILOTING: logFile.setLevel(prefs.piloting['pilotLoggingLevel'])
    else: logFile.setLevel(logging.getLevel('exp'))
    return logFile

def setupWindow(expInfo=None, win=None):
    if PILOTING: logging.debug('Fullscreen settings ignored as running in pilot mode.')
    if win is None:
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0, winType='pyglet', allowGUI=True,
            allowStencil=False, monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none', blendMode='avg', useFBO=True,
            units='height', checkTiming=False 
        )
    else:
        win.color = [0,0,0]; win.colorSpace = 'rgb'; win.backgroundImage = ''
        win.backgroundFit = 'none'; win.units = 'height'
    if expInfo is not None:
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        if prefs.piloting['showPilotingIndicator']: win.showPilotingIndicator()
        if prefs.piloting['forceMouseVisible']: win.mouseVisible = True
    return win

def setupDevices(expInfo, thisExp, win):
    ioConfig = {'Keyboard': dict(use_keymap='psychopy'), 'Experiment': dict(filename=thisExp.dataFileName)}
    ioServer = io.launchHubServer(window=win, **ioConfig)
    deviceManager.ioServer = ioServer
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub')
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    if thisExp.status != PAUSED: return
    pauseTimer = core.Clock()
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents(): comp.pause()
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(deviceName='defaultKeyboard', backend='ioHub') # simplified
    while thisExp.status == PAUSED:
        if defaultKeyboard.getKeys(keyList=['escape']): endExperiment(thisExp, win=win)
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents(): comp.device.dispatchMessages()
        clock.time.sleep(0.001)
    if thisExp.status == FINISHED: endExperiment(thisExp, win=win)
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents(): comp.play()
    for timer in timers: timer.addTime(-pauseTimer.getTime())

# --- Helper function for component timing ---
def manage_component_timing(component, t_routine, tThisFlipGlobal, frameN, win, thisExp,
                            start_time=0.0, duration=None, end_on_duration=True, component_name_log=None):
    """Manages start, drawing, and stop of a component for a fixed duration or until externally stopped."""
    if component_name_log is None:
        component_name_log = component.name if hasattr(component, 'name') else 'unnamed_component'
    
    frameTolerance = 0.001 # How close to onset before 'same' frame ( लिया from original)

    if component.status == NOT_STARTED and t_routine >= start_time - frameTolerance:
        component.frameNStart = frameN
        component.tStart = t_routine # Local routine time
        component.tStartRefresh = tThisFlipGlobal
        win.timeOnFlip(component, 'tStartRefresh')
        thisExp.timestampOnFlip(win, f'{component_name_log}.started')
        component.status = STARTED
        component.setAutoDraw(True)

    if component.status == STARTED:
        if duration is not None and end_on_duration:
            if tThisFlipGlobal > component.tStartRefresh + duration - frameTolerance:
                component.tStop = t_routine # Local routine time
                component.frameNStop = frameN
                component.tStopRefresh = tThisFlipGlobal
                thisExp.timestampOnFlip(win, f'{component_name_log}.stopped')
                component.status = FINISHED
                component.setAutoDraw(False)
    return component.status

def create_rule(rule, n_trials=50):
    """
    Generate a list of n_trials trial dicts for the given WCST rule.
    Each dict has keys:
        - 'number':    1, 2, 3 or 4
        - 'color':     'red', 'green', 'yellow', or 'blue'
        - 'shape':     'triangle', 'star', 'cross', or 'dot'
        - 'corrAns':   the name of the one reference card that is correct 
                       under the specified rule

    The four immutable "reference cards" are defined as:
        • one_red_dot         (number=1, color='red',    shape='Dot')
        • two_yellow_triangle (number=2, color='yellow', shape='Triangle')
        • three_green_cross   (number=3, color='green',  shape='Cross')
        • four_blue_star      (number=4, color='blue',   shape='Star')

    Parameters:
    -----------
    rule : str
        One of {'color', 'shape', 'number'}.  Determines which attribute 
        is the “sorting rule” on these trials.
    n_trials : int
        How many random combinations to return (default=50).  Must be ≤ 64.

    Returns:
    --------
    trials : list of dict
        A list of length n_trials.  Each element is a dict with keys 
        'number', 'color', 'shape', and 'corrAns'.  The value of 'corrAns' 
        is one of:
            'one_red_dot', 'two_yellow_triangle', 'three_green_cross', 'four_blue_star'
    """

    # 1) Define the four reference cards (the “correct answers” for each dimension).
    #    Note: shapes are capitalized to match your filenames (e.g. "Dot", "Triangle", etc.).
    reference_cards = [
        { 'name': 'one_red_dot',         'number': 1, 'color': 'red',   'shape': 'Dot'      },
        { 'name': 'two_yellow_triangle', 'number': 2, 'color': 'yellow','shape': 'Triangle' },
        { 'name': 'three_green_cross',   'number': 3, 'color': 'green', 'shape': 'Cross'    },
        { 'name': 'four_blue_star',      'number': 4, 'color': 'blue',  'shape': 'Star'     }
    ]

    # 2) All possible attribute values:
    all_numbers = [1, 2, 3, 4]
    all_colors  = ['red', 'green', 'yellow', 'blue']
    all_shapes  = ['triangle', 'star', 'cross', 'dot']

    # 3) Build the full set of 4×4×4 = 64 possible cards:
    all_combinations = [
        { 'number': num, 'color': col, 'shape': shp }
        for num, col, shp in itertools.product(all_numbers, all_colors, all_shapes)
    ]

    # 4) Randomly sample n_trials distinct cards from that pool:
    if n_trials > len(all_combinations):
        raise ValueError(f"n_trials={n_trials} is too large; max is 64.")

    # ───── Use the standard‐library random.sample ─────
    import random as _py_random
    sampled = _py_random.sample(all_combinations, k=n_trials)
    # ──────────────────────────────────────────────────

    # 5) Tag each sampled card with exactly one 'corrAns', depending on rule:
    for trial in sampled:
        if rule == 'color':
            this_color = trial['color']
            match = next(ref for ref in reference_cards if ref['color'] == this_color)

        elif rule == 'number':
            this_num = trial['number']
            match = next(ref for ref in reference_cards if ref['number'] == this_num)

        elif rule == 'shape':
            # Capitalize the lowercase shape string so it matches reference_cards['shape']
            this_shape_cap = trial['shape'].capitalize()  # e.g. 'dot' → 'Dot'
            match = next(ref for ref in reference_cards if ref['shape'] == this_shape_cap)

        else:
            raise ValueError("rule must be one of {'color', 'shape', 'number'}.")

        trial['corrAns'] = match['name']

    return sampled



def run_trials_dynamic(win,
                       thisExp,
                       stims_for_response_phase,
                       choice_cards_list,
                       response_mouse,
                       fixation,
                       feedback_text,
                       globalClock,
                       routineTimer,
                       defaultKeyboard,
                       initial_rule=None,
                       criterion=10,
                       n_trials_per_block=64,
                       max_runs=6):
    """
    Runs WCST trials with dynamic rule switching and writes rich per-trial logs.

    Now logs to PsychoPy .log:
      - Each trial: TRIAL, rule, image, clicked, correct, rt, abs_time
      - Rule change: RULE_CHANGE with abs_time, trial index, from→to, run_count

    Also writes optional trial attributes into the CSV (can be removed if undesired).
    """

    rule_list = ['color', 'shape', 'number']
    current_rule = initial_rule if initial_rule in rule_list else random.choice(rule_list)

    def pick_different_rule(prev_rule):
        others = [r for r in rule_list if r != prev_rule]
        return random.choice(others)

    consecutive_correct = 0
    run_count = 0

    batch_trials = create_rule(current_rule, n_trials=n_trials_per_block)
    trial_index_in_batch = 0

    # Count only actually-presented trials (skip missing files)
    trials_shown = 0

    IMAGE_FOLDER = 'images'
    IMAGE_EXT    = '.jpg'

    while True:
        # Refresh batch if needed
        if trial_index_in_batch >= len(batch_trials):
            batch_trials = create_rule(current_rule, n_trials=n_trials_per_block)
            trial_index_in_batch = 0

        # Pull the next trial definition
        trial = batch_trials[trial_index_in_batch]
        trial_index_in_batch += 1

        image_fname = os.path.join(
            IMAGE_FOLDER,
            f"{trial['number']}{trial['color']}{trial['shape'].capitalize()}{IMAGE_EXT}"
        )

        # Skip if image missing
        if not os.path.isfile(image_fname):
            logging.warning(f"Missing trial‐card image, skipping: {image_fname}")
            continue

        # We will present this trial
        trials_shown += 1

        # Present the stimulus
        stims_for_response_phase['trial_card'].setImage(image_fname)
        response_mouse.clicked_card_name_str = None
        response_mouse.rt = 0

        # Absolute time stamp for the row + for the log
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        abs_t = globalClock.getTime()  # float seconds

        # ---- Fixation ----
        if not run_fixation_phase(win, thisExp, fixation, routineTimer, globalClock, defaultKeyboard, duration=1.0):
            return False  # Esc pressed

        # ---- Response phase (show choice + trial card, wait click) ----
        for choice_stim in choice_cards_list:
            choice_stim.setAutoDraw(True)

        continue_exp, clicked_name, rt = run_response_phase(
            win, thisExp, stims_for_response_phase, response_mouse, routineTimer, globalClock, defaultKeyboard
        )
        if not continue_exp:
            return False

        # Raw mouse data
        mx, my = response_mouse.getPos()
        mbuttons = response_mouse.getPressed()
        thisExp.addData('response.x', mx)
        thisExp.addData('response.y', my)
        thisExp.addData('response.leftButton', mbuttons[0])
        thisExp.addData('response.midButton', mbuttons[1])
        thisExp.addData('response.rightButton', mbuttons[2])
        thisExp.addData('response.rt', rt)

        # Evaluate correctness
        correct_answer_name = trial['corrAns']
        score = 1 if (clicked_name is not None and clicked_name == correct_answer_name) else 0
        thisExp.addData('response.corr', score)
        thisExp.addData('response.clicked_name', clicked_name if clicked_name is not None else '')

        # ---- Optional: also save rule + trial attributes to CSV ----
        thisExp.addData('trial.index', trials_shown)
        thisExp.addData('trial.rule', current_rule)
        thisExp.addData('trial.number', trial['number'])
        thisExp.addData('trial.color', trial['color'])
        thisExp.addData('trial.shape', trial['shape'])
        thisExp.addData('trial.corrAns', correct_answer_name)
        thisExp.addData('trial.image', image_fname)

        # ---- Feedback ----
        if not run_feedback_phase(win, thisExp, score, feedback_text, routineTimer, globalClock, defaultKeyboard, duration=1.0):
            return False

        # ---- Per-trial LOG entry ----
        # Use EXP level so it appears with your current log level.
        try:
            logging.exp(
                f"TRIAL n={trials_shown} rule={current_rule} image='{os.path.basename(image_fname)}' "
                f"clicked='{clicked_name}' correct={score} rt={rt:.3f} abs_time={abs_t:.6f}"
            )
        except Exception:
            pass

        thisExp.nextEntry()

        # Update streak toward criterion
        if score == 1:
            consecutive_correct += 1
        else:
            consecutive_correct = 0

        # Completed a run?
        if consecutive_correct >= criterion:
            run_count += 1

            # Decide whether to end
            if run_count >= max_runs:
                try:
                    logging.exp(
                        f"RUN_COMPLETE count={run_count} criterion={criterion} "
                        f"final_trial={trials_shown} abs_time={globalClock.getTime():.6f}"
                    )
                except Exception:
                    pass
                return True

            # Switch rule
            prev_rule = current_rule
            current_rule = pick_different_rule(prev_rule)
            consecutive_correct = 0

            # Trigger for rule change (already present)
            send_trigger(4)

            # Log the RULE CHANGE with time & trial number
            try:
                logging.exp(
                    f"RULE_CHANGE trial={trials_shown} from={prev_rule} to={current_rule} "
                    f"run_count={run_count} abs_time={globalClock.getTime():.6f}"
                )
            except Exception:
                pass

            # Fresh batch for the new rule
            batch_trials = create_rule(current_rule, n_trials=n_trials_per_block)
            trial_index_in_batch = 0




# --- New Trial Phase Functions ---
def initialize_trial_globals(thisTrial_data, trial_card_stim, mouse_obj):
    """Sets card image, correct answer, resets mouse state for the trial."""
    trial_card_stim.setImage(thisTrial_data['card']) # 'card' from cards.xlsx
    mouse_obj.clicked_card_name_str = None # Store the name of the clicked card as a string
    mouse_obj.rt = 0 # Reaction time for this trial
    
    # For check_for_click's internal debouncing, reset if it uses mouse-specific state that needs per-trial reset
    # However, check_for_click as provided should handle its own state per mouse object.
    # If mouse_obj.prevButtonState was used, it would be reset here.
    # mouse_obj.prev_button_state_for_trial = mouse_obj.getPressed() # Example if custom debouncing

    print(f"Trial Start: Card Image: {thisTrial_data['card']}, Correct Answer ID: {thisTrial_data['corrAns']}")
    return thisTrial_data['corrAns'] # Return corrAns for clarity


def evaluate_response(clicked_name, correct_answer_name, thisExp):
    corr = 0
    if clicked_name is not None and clicked_name == correct_answer_name:
        corr = 1
    thisExp.addData('response.corr', corr)
    thisExp.addData('response.clicked_name', clicked_name if clicked_name is not None else '')
    print(f"Evaluation: Clicked='{clicked_name}', Correct='{correct_answer_name}', Score={corr}")
    return corr

def run_response_phase(win, thisExp, stims, mouse_obj, routineTimer, globalClock, defaultKeyboard):
    """
    Displays all four choice cards (top row) and the trial card (bottom) at once.
    Waits for a click on any of the choice cards.
    Returns (True, clicked_name, rt). If Esc is pressed, returns (False, None, 0).
    """

    trial_card_stim = stims['trial_card']
    choice_cards_stims = stims['choice_cards']

    # 1) Reset statuses and turn AutoDraw OFF for everything
    trial_card_stim.status = NOT_STARTED
    trial_card_stim.setAutoDraw(False)
    for card_s in choice_cards_stims:
        card_s.status = NOT_STARTED
        card_s.setAutoDraw(False)

    mouse_obj.status = NOT_STARTED
    mouse_obj.clicked_card_name_str = None
    mouse_obj.rt = 0

    clicked_name = None
    first_frame = True

    # 2) Start the routine‐timer
    routineTimer.reset()
    frameN = -1
    continueRoutine = True

    while continueRoutine:
        t_routine = routineTimer.getTime()
        tThisFlipGlobal = win.getFutureFlipTime(clock=globalClock)
        frameN += 1

        # On the very first frame (frameN == 0), draw ALL five cards at once:
        if first_frame:
            # --- Draw the trial card ---
            trial_card_stim.setAutoDraw(True)
            trial_card_stim.status = STARTED
            
            # --- REMOVED ---
            # The trigger for card presentation has been removed.
            win.callOnFlip(send_trigger, 1)

            # --- Draw all four choice cards (top row) ---
            for card_s in choice_cards_stims:
                card_s.setAutoDraw(True)
                card_s.status = STARTED

            # --- Start the mouse right away at t=0 ---
            mouse_obj.status = STARTED
            mouse_obj.mouseClock.reset()
            thisExp.addData('mouse.response_phase_started_rt', t_routine)

            first_frame = False

        # 3) If mouse is active, check for clicks:
        if mouse_obj.status == STARTED:
            
            # --- MODIFIED ---
            # The trigger_map has been removed. We now call check_for_click
            # without it, as it will no longer send triggers.
            clicked_name = check_for_click(mouse_obj, choice_cards_stims)
            
            if clicked_name is not None:
                mouse_obj.clicked_card_name_str = clicked_name
                mouse_obj.rt = mouse_obj.mouseClock.getTime()
                print(f"Clicked: {mouse_obj.clicked_card_name_str}, RT: {mouse_obj.rt:.3f}s")
                continueRoutine = False

        # 4) Check for Escape key (quit immediately)
        if len(event.getKeys(keyList=["escape"])) > 0:
            return False, None, 0

        # 5) Break if someone clicked
        if not continueRoutine:
            break

        # 6) Flip the window for the next frame
        win.flip()

    # 7) After a click: hide the trial card so it doesn't persist
    if trial_card_stim.status == STARTED:
        trial_card_stim.setAutoDraw(False)
        trial_card_stim.status = FINISHED

    return True, mouse_obj.clicked_card_name_str, mouse_obj.rt



def run_fixation_phase(win, thisExp, fixation_stim, routineTimer, globalClock, defaultKeyboard, duration=1.0):
    """
    Displays a fixation cross for duration seconds.
    Returns False immediately if Escape is pressed, otherwise True on normal completion.
    """
    continueRoutine = True
    fixation_stim.status = NOT_STARTED

    routineTimer.reset()
    frameN = -1
    while continueRoutine:
        t_routine = routineTimer.getTime()
        tThisFlipGlobal = win.getFutureFlipTime(clock=globalClock)
        frameN += 1

        # Start drawing the fixation cross at t=0
        manage_component_timing(
            fixation_stim, t_routine, tThisFlipGlobal, frameN, win, thisExp,
            start_time=0.0, duration=duration, component_name_log='fixation'
        )
        if fixation_stim.status == FINISHED:
            continueRoutine = False

        # Check for Escape (using event.getKeys)
        if len(event.getKeys(keyList=["escape"])) > 0:
            return False  # signal to end experiment immediately

        if not continueRoutine:
            break
        win.flip()

    # Ensure that fixation is turned off if we exit early
    if fixation_stim.status == STARTED:
        fixation_stim.setAutoDraw(False)
        fixation_stim.status = FINISHED

    return True


def run_feedback_phase(win, thisExp, score, feedback_stim, routineTimer, globalClock, defaultKeyboard, duration=1.0):
    """
    Displays 'Correct!' or 'Incorrect' for duration seconds.
    Returns False immediately if Escape is pressed; otherwise True on normal completion.
    Sends trigger=66 if correct, trigger=68 if incorrect.
    """
    msg = "Correct!" if score == 1 else "Incorrect"
    if score == 1:
        send_trigger(2)
    else:
        send_trigger(3)

    feedback_stim.setText(msg)
    feedback_stim.status = NOT_STARTED

    routineTimer.reset()
    frameN = -1
    continueRoutine = True

    while continueRoutine:
        t_routine = routineTimer.getTime()
        tThisFlipGlobal = win.getFutureFlipTime(clock=globalClock)
        frameN += 1

        manage_component_timing(
            feedback_stim, t_routine, tThisFlipGlobal, frameN, win, thisExp,
            start_time=0.0, duration=duration, component_name_log='feedback_text'
        )

        if feedback_stim.status == FINISHED:
            continueRoutine = False

        # Check for Escape (using event.getKeys)
        if len(event.getKeys(keyList=["escape"])) > 0:
            return False

        if not continueRoutine:
            break
        win.flip()

    # Ensure feedback is turned off if we exit early
    if feedback_stim.status == STARTED:
        feedback_stim.setAutoDraw(False)
        feedback_stim.status = FINISHED

    return True

def display_end_screen(screen, clock):
    """
    Displays a German end-of-block message and waits for the user to press SPACE.

    Args:
        screen: The main pygame.Surface object to draw on.
        clock: The main pygame.time.Clock object to control the framerate.
    """
    # Define fonts and colors for the screen
    font_title = pygame.font.Font(None, 60)
    font_prompt = pygame.font.Font(None, 40)
    white = (255, 255, 255)
    black = (0, 0, 0)

    # Prepare the text surfaces
    title_surface = font_title.render("Block Beendet", True, black)
    prompt_surface = font_prompt.render("Drücken Sie die Leertaste, um die Pause zu beginnen.", True, black)

    # Center the text on the screen
    screen_width = screen.get_width()
    screen_height = screen.get_height()
    title_rect = title_surface.get_rect(center=(screen_width / 2, screen_height / 2 - 40))
    prompt_rect = prompt_surface.get_rect(center=(screen_width / 2, screen_height / 2 + 30))

    # Event loop: waits for the spacebar or quit event
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False # Exit the loop

        # Drawing logic
        screen.fill(white)
        screen.blit(title_surface, title_rect)
        screen.blit(prompt_surface, prompt_rect)
        pygame.display.flip()

        # Maintain a reasonable framerate
        clock.tick(30)

# =====================================================================================
# MAIN RUN FUNCTION 
# =====================================================================================

def run_wcst(win, io, participant_id, **kwargs): # MODIFIED: New function signature
    """
    Modular, crash-resistant WCST wrapper for chained tasks.
    Accepts a pre-existing window and ioHub server.

    Args:
        win (psychopy.visual.Window): The main experiment window.
        io (psychopy.iohub.client.ioHubConnection): The active ioHub server instance.
        participant_id (str): The participant ID.
        **kwargs: Catches any other arguments the main script might send.
    """
    print(f"--- Starting Wisconsin Card Sorting Task for participant {participant_id} ---")

    # The deviceManager is a global singleton, so we can check for the ioServer
    # that the main script already launched.
    deviceManager.ioServer = io
    
    # Fresh per-call expInfo (don't mutate module global)
    expInfo_local = expInfo.copy()
    expInfo_local["participant"] = participant_id # MODIFIED: Use the passed-in participant_id
    expInfo_local["date"] = data.getDateStr()

    thisExp = None
    completed = False
    
    # This try...finally block ensures that even if the task crashes,
    # the experiment can continue, and data is saved.
    try:
        # --- Setup Data and Logging (specific to this task) ---
        # This part is still needed to create wcst-specific data files.
        thisExp = setupData(expInfo=expInfo_local)
        logFile = setupLogging(filename=thisExp.dataFileName) # Optional but good practice
        configure_logging_for_trials('exp')  # or 'data'

        # --- Run the core task logic ---
        # We pass the SHARED window 'win' to the main run function.
        run(
            expInfo=expInfo_local,
            thisExp=thisExp,
            win=win, # MODIFIED: Use the passed-in window
            globalClock=core.Clock(format="float"),
        )
        completed = True
        print("--- WCST Completed Successfully ---")

    except SystemExit:
        # This allows a hard quit (like pressing Escape) to stop everything.
        print("--- WCST aborted by user (SystemExit) ---")
        raise

    except Exception as e:
        # Catches any other crash and logs it, preventing the main script from stopping.
        print(f"CRITICAL ERROR in WCST module: {e}")
        import traceback
        traceback.print_exc() # Prints detailed error info
        if thisExp is not None:
            thisExp.addData('wcst_wrapper_error', str(e))
            saveData(thisExp)

    finally:
        # --- Cleanup ---
        # We do NOT close the window or quit ioHub here, as the main script owns them.
        # We just make sure the data for this task has been saved.
        if thisExp is not None and thisExp.status != FINISHED:
             saveData(thisExp)
        
        print("--- Exiting WCST module ---")

    return completed






def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    thisExp.status = STARTED
    win.winHandle.activate()
    ioServer = deviceManager.ioServer
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub'
        )
    if defaultKeyboard.getKeys(keyList=["escape"]):
        endExpNow = True
    else:
        endExpNow = False

    os.chdir(_thisDir)
    filename = thisExp.dataFileName
    frameTolerance = 0.001 
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0 

    # --- Initialize components for Routine "Instructions" ---
    instructions = visual.TextStim(
        win=win, name='instructions',
        text="Ihre Aufgabe ist es, jede Karte, die unten erscheint, einer der vier Karten oben zuzuordnen. Nach jeder Zuordnung erfahren Sie, ob Ihre Antwort 'richtig' oder 'falsch' war. Nutzen Sie dieses Feedback, um die geltende Sortierregel herauszufinden.",
        font='Arial', pos=(0, 0), draggable=False, height=0.05, wrapWidth=.9, ori=0, 
        color='white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0,
        
    )
    continue_button = visual.ImageStim(
        win=win, name='continue_button', image='continue.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -.4), draggable=False, size=(0.65, .1), color=[1,1,1], 
        colorSpace='rgb', opacity=None, flipHoriz=False, flipVert=False, texRes=128.0, 
        interpolate=True, depth=-1.0
    )
    mouse_2 = event.Mouse(win=win)  # For instructions
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "Example" ---
    example_text = visual.TextStim(
        win=win, name='example_text',
        text='Zum Beispiel werden Sie einen Bildschirm wie diesen sehen:',
        font='Arial', pos=(0, 0.4), draggable=False, height=0.04, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0
    )
    example_image = visual.ImageStim(
        win=win, name='example_image', image='example.jpg', mask=None, anchor='center',
        ori=0, pos=(0, 0.15), draggable=False, size=(0.7, 0.4), color=[1,1,1], colorSpace='rgb', 
        opacity=1, flipHoriz=False, flipVert=False, texRes=128, interpolate=True, depth=-1.0
    )
    example_text_2 = visual.TextStim(
        win=win, name='example_text_2',
        text="Klicken Sie auf eine der vier oberen Karten, von der Sie denken, dass sie zur unteren Karte passt. Nach Ihrer Wahl wird Ihnen mitgeteilt, ob Sie 'richtig' oder 'falsch' lagen. Nutzen Sie dieses Feedback, um die Regel zu lernen. Klicken Sie auf das Bild, um das Experiment zu starten.",
        font='Arial', pos=(0, -0.25), draggable=False, height=0.04, wrapWidth=.9, ori=0, 
        color='white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=-2.0
    )
    mouse = event.Mouse(win=win)  # For example screen
    mouse.mouseClock = core.Clock()
    
    one_red_dot = visual.ImageStim(
        win=win, name='one_red_dot',
        image='images/1redDot.jpg',
        ori=0, pos=(-0.375, 0.25), size=(0.2, 0.2)
    )
    two_yellow_triangle = visual.ImageStim(
        win=win, name='two_yellow_triangle',
        image='images/2yellowTriangle.jpg',
        ori=0, pos=(-0.125, 0.25), size=(0.2, 0.2)
    )
    three_green_cross = visual.ImageStim(
        win=win, name='three_green_cross',
        image='images/3greenCross.jpg',
        ori=0, pos=(0.125, 0.25), size=(0.2, 0.2)
    )
    four_blue_star = visual.ImageStim(
        win=win, name='four_blue_star',
        image='images/4blueStar.jpg',
        ori=0, pos=(0.375, 0.25), size=(0.2, 0.2)
    )

    # --- Create the trial_card stimulus here, BEFORE you hand it to the trial loops ---
    #    This is the crucial addition: now trial_card_stim exists, so run_trials_dynamic
    #    (and initialize_trial_globals) can call trial_card_stim.setImage(...) each trial.
    trial_card_stim = create_trial_card_stim(
        win=win,
        initial_image='images/default.jpg',
        pos=(0, -0.2),
        size=(0.25, 0.25)
    )

    # Master mouse object for trials
    response_mouse = event.Mouse(win=win, visible=True)
    response_mouse.mouseClock = core.Clock()
    response_mouse.clicked_card_name_str = None 
    response_mouse.rt = 0

    # Store choice cards in a list for convenience
    choice_cards_list = [
        one_red_dot,
        two_yellow_triangle,
        three_green_cross,
        four_blue_star
    ]

    # Pass our freshly created trial_card_stim into the dict that run_trials_dynamic expects
    stims_for_response_phase = {
        'trial_card': trial_card_stim,
        'choice_cards': choice_cards_list
    }

    # --- Initialize components for Routine "Feedback" ---
    feedback_text = visual.TextStim(
        win=win, name='feedback_text', text='', font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=-1.0
    )
        
        # --- Create a proper fixation cross stimulus (so run_fixation_phase can use it) ---
    fixation = visual.TextStim(
        win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0),
        height=0.1,
        color='white',
        colorSpace='rgb'
    )
    
    # --- Initialize components for Routine "End" ---
    thank_you = visual.TextStim(
        win=win, name='thank_you',
        text='Dies ist das Ende des Aufgabe.',
        font='Arial', pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0
    )

    # Clocks
    if globalClock is None:
        globalClock = core.Clock()
    elif isinstance(globalClock, str):
        if globalClock == 'float':
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()
    win.flip()
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z',
        fractionalSecondDigits=6
    )

    # --- Run Routine "Instructions" ---
    Instructions = data.Routine(
        name='Instructions',
        components=[instructions, continue_button, mouse_2]
    )
    send_trigger(0)  # WCST Task Instructions Start
    continueRoutine = True
    instructions.status = continue_button.status = mouse_2.status = NOT_STARTED
    routineTimer.reset()
    frameN = -1
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=globalClock)
        frameN += 1
        
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            instructions.frameNStart = frameN
            instructions.tStart = t
            instructions.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(instructions, 'tStartRefresh')
            thisExp.timestampOnFlip(win, 'instructions.started')
            instructions.status = STARTED
            instructions.setAutoDraw(True)
            
        if continue_button.status == NOT_STARTED and tThisFlip >= 1.0 - frameTolerance:
            continue_button.frameNStart = frameN
            continue_button.tStart = t
            continue_button.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(continue_button, 'tStartRefresh')
            thisExp.timestampOnFlip(win, 'continue_button.started')
            continue_button.status = STARTED
            continue_button.setAutoDraw(True)

        if mouse_2.status == NOT_STARTED and t >= 1.0 - frameTolerance:
            mouse_2.frameNStart = frameN
            mouse_2.tStart = t
            mouse_2.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(mouse_2, 'tStartRefresh')
            thisExp.addData('mouse_2.started', t)
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()
            
        if mouse_2.status == STARTED:
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:
                prevButtonState = buttons
                if sum(buttons) > 0:
                    if continue_button.contains(mouse_2):
                        continueRoutine = False
            
        if defaultKeyboard.getKeys(keyList=["escape"]):
            endExpNow = True
        if endExpNow or not continueRoutine:
            break
        if thisExp.status == PAUSED:
            pauseExperiment(thisExp, win, [routineTimer, globalClock], Instructions)
        win.flip()
    
    for comp in Instructions.components:
        if hasattr(comp, 'setAutoDraw'):
            comp.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    routineTimer.reset()
    if endExpNow:
        endExperiment(thisExp, win)
        return

    # --- Run Routine "Example" ---
    Example = data.Routine(
        name='Example',
        components=[example_text, example_image, example_text_2, mouse]
    )
    continueRoutine = True
    example_text.status = example_image.status = example_text_2.status = mouse.status = NOT_STARTED
    routineTimer.reset()
    frameN = -1
    while continueRoutine:
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=globalClock)
        frameN += 1

        if example_text.status == NOT_STARTED:
            example_text.status = STARTED
            example_text.setAutoDraw(True)
        if example_image.status == NOT_STARTED:
            example_image.status = STARTED
            example_image.setAutoDraw(True)
        if example_text_2.status == NOT_STARTED:
            example_text_2.status = STARTED
            example_text_2.setAutoDraw(True)
        
        if mouse.status == NOT_STARTED and t >= 1.0 - frameTolerance:
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = mouse.getPressed()
        
        if mouse.status == STARTED:
            buttons = mouse.getPressed()
            if buttons != prevButtonState:
                prevButtonState = buttons
                if sum(buttons) > 0:
                    if example_image.contains(mouse):
                        continueRoutine = False
        
        if defaultKeyboard.getKeys(keyList=["escape"]):
            endExpNow = True
        if endExpNow or not continueRoutine:
            break
        if thisExp.status == PAUSED:
            pauseExperiment(thisExp, win, [routineTimer, globalClock], Example)
        win.flip()

    for comp in Example.components:
        if hasattr(comp, 'setAutoDraw'):
            comp.setAutoDraw(False)
    thisExp.addData('Example.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    routineTimer.reset()
    if endExpNow:
        endExperiment(thisExp, win)
        return

    # -------------------------------------------------------
    # Run the entire experiment with dynamic rule-switching:
    # -------------------------------------------------------
    success = run_trials_dynamic(
        win=win,
        thisExp=thisExp,
        stims_for_response_phase=stims_for_response_phase,
        choice_cards_list=choice_cards_list,
        response_mouse=response_mouse,
        fixation=fixation,
        feedback_text=feedback_text,
        globalClock=globalClock,
        routineTimer=routineTimer,
        defaultKeyboard=defaultKeyboard,
        initial_rule='color',   # or None to randomize at start
        criterion=10,           # “10” correct in a row before switching
        n_trials_per_block=64   # number of create_rule calls per batch
    )
    if not success:
        endExpNow = True



    # --- Run Routine "End" ---
    End = data.Routine(name='End', components=[thank_you])
    continueRoutine = True
    thank_you.status = NOT_STARTED
    routineTimer.reset()
    frameN = -1
    end_duration = 3.0
    while continueRoutine and routineTimer.getTime() < end_duration:
        t = routineTimer.getTime()
        tThisFlipGlobal = win.getFutureFlipTime(clock=globalClock)
        frameN += 1
        if thank_you.status == NOT_STARTED:
            thank_you.status = STARTED
            thank_you.setAutoDraw(True)
            thisExp.addData('thank_you.started', tThisFlipGlobal)

        if defaultKeyboard.getKeys(keyList=["escape"]):
            endExpNow = True
        if endExpNow:
            break
        if thisExp.status == PAUSED:
            pauseExperiment(thisExp, win, [routineTimer, globalClock], End)
        win.flip()

    if hasattr(thank_you, 'setAutoDraw'):
        thank_you.setAutoDraw(False)
    thisExp.addData('End.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()

    win.flip()
    endExperiment(thisExp, win)



def saveData(thisExp):
    filename = thisExp.dataFileName
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)
    logging.flush()

def endExperiment(thisExp, win=None):
    if win is not None:
        win.clearAutoDraw()
        win.flip()
    logging.console.setLevel(logging.WARNING)
    thisExp.status = FINISHED
    for fcn in runAtExit: fcn()
    logging.flush()

def quit(thisExp, win=None, thisSession=None):
    saveData(thisExp) # Ensure data is saved on quit
    thisExp.abort() 
    if win is not None:
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None: thisSession.stop()
    core.quit()

# --- Main script execution ---
if __name__ == '__main__':
    # --- Get Participant Info via Dialog Box ---
    try:
        # 💡 FIX: Do NOT redefine expInfo here.
        # The global expInfo dictionary already exists and contains the date.
        # We just need to show the dialog box to fill in the participant ID.
        
        # OLD CODE (incorrect):
        # expInfo = {'participant': ''} 
        # expInfo = showExpInfoDlg(expInfo=expInfo)

        # NEW CODE (correct):
        expInfo = showExpInfoDlg(expInfo=expInfo) # Directly use the global expInfo
        
        if expInfo is None: # Check if the dialog was cancelled
            core.quit()
            
    except Exception as e:
        print(f"Failed to show experiment info dialog: {e}")
        core.quit()

    # --- Setup Experiment ---
    # (The rest of your code from here is correct)
    thisExp = None
    win = None
    
    try:
        print(f"Running with participant ID from dialog: {expInfo['participant']}")
        thisExp = setupData(expInfo=expInfo)
        logFile = setupLogging(filename=thisExp.dataFileName)
        win = setupWindow(expInfo=expInfo)
        setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
        
        run(expInfo=expInfo, thisExp=thisExp, win=win, globalClock='float')

    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}")
        import traceback
        traceback.print_exc() # This will give you a more detailed error message in the future
        if thisExp is not None:
            thisExp.addData('experiment_error', str(e))
            saveData(thisExp)
        raise

    finally:
        print("Experiment ended. Performing cleanup.")
        if thisExp is not None and thisExp.status != FINISHED:
              quit(thisExp=thisExp, win=win)
        elif win is not None:
            win.close()
        
        core.quit()