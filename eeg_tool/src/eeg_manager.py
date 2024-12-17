import numpy as np
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import os
import time
import random
from mne.decoding import CSP

class EEGManager:
    def __init__(self):
        # Board setup
        self.board = None
        self.is_recording = False
        self.is_streaming = False
        
        # Session management
        self.current_session_num = None
        self.next_session_number = self._get_next_session_number()
        
        # File management
        self.data_dir = 'data'
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.formatted_dir = os.path.join(self.data_dir, 'formatted')
        self.annotation_file = os.path.join(self.data_dir, 'annotations.txt')
        
        # Create directories
        for directory in [self.raw_dir, self.processed_dir, self.formatted_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Channel setup
        self.channel_states = {
            'FCz': True,
            'C3': True,
            'Cz': True,
            'CPz': True,
            'C2': True,
            'C4': True
        }
        self.channel_names = list(self.channel_states.keys())  # Store current channel names
        
        # Task setup
        self.task_names = {
            "Right_Hand": ["Imagine or perform right hand movement", 21],
            "Left_Hand": ["Imagine or perform left hand movement", 21],
            "Blinking": ["Blink your eyes at a comfortable pace", 21],
            "Jaw_Clenching": ["Clench your jaw with moderate force", 21],
            "Relax": ["Relax your body and mind, do nothing", 21]
        }
        self.task_sequence = []
        self.current_task_index = 0
        self.task_duration = 5  # Fixed at 5 seconds
        self.break_duration = 3  # Default break duration
        
        # Initialize board
        self.initialize_board()
        
        # Initialize session counter
        self.next_session_number = self._get_next_session_number()
        
        # Data storage
        self.raw_data = None
        
    def _get_next_session_number(self):
        """Determine the next available session number from annotations file"""
        try:
            if not os.path.exists(self.annotation_file):
                return 1
            
            with open(self.annotation_file, 'r') as f:
                content = f.read()
                sessions = [int(x.replace('session', '')) 
                          for x in content.split('\n') 
                          if x.strip().startswith('session')]
                return max(sessions, default=0) + 1
        except Exception as e:
            print(f"Error reading session number: {e}")
            return 1
        
    def initialize_board(self, port='COM9'):
        """Initialize OpenBCI board"""
        try:
            if self.is_streaming:
                return True
            
            params = BrainFlowInputParams()
            params.serial_port = port
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            self.board.prepare_session()
            self.board.start_stream()
            self.is_streaming = True
            print("Board initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing board: {e}")
            self.is_streaming = False
            return False
            
    def stop_stream(self):
        """Stop board streaming"""
        if self.board and self.is_streaming:
            try:
                self.board.stop_stream()
                self.board.release_session()
                self.board = None
                self.is_streaming = False
                print("Board streaming stopped")
                return True
            except Exception as e:
                print(f"Error stopping board: {e}")
                return False
        return True
        
    def initialize_session(self, session_num):
        """Initialize a new recording session"""
        self.current_session_num = self.next_session_number
        self.task_sequence = []
        self.current_task_index = 0
        
        # Create balanced task sequence
        for task_name, (instruction, count) in self.task_names.items():
            self.task_sequence.extend([(task_name, instruction)] * count)
            
        # Randomize task order
        random.shuffle(self.task_sequence)
        
        # Write session header to annotations file
        with open(self.annotation_file, "a+") as f:
            f.write(f"\nsession{self.current_session_num}\n")
        
        # Increment session number for next time
        self.next_session_number += 1
        return True
        
    def get_next_task(self):
        """Get next task in sequence"""
        if self.current_task_index < len(self.task_sequence):
            task = self.task_sequence[self.current_task_index]
            self.current_task_index += 1
            return task
        return None
        
    def start_recording(self):
        """Start recording current task"""
        if self.board and self.is_streaming:
            self.is_recording = True
            self.current_eeg_data = []
            self.board.get_board_data()  # Clear buffer
            
            # Record task in annotations
            if self.current_task_index > 0:
                task_name = self.task_sequence[self.current_task_index - 1][0]
                with open(self.annotation_file, "a+") as f:
                    task = task_name.split('_')[0]
                    if task == "Jaw":
                        task = "Jaw_Clenching"
                    elif task in ["Left", "Right"]:
                        task = f"{task}_Hand"
                    f.write(f"{task}\n")
                    
            return True
        return False
        
    def stop_recording(self):
        """Stop recording and save data"""
        if self.is_recording:
            self.is_recording = False
            
            # Get recorded data
            data = self.board.get_board_data()
            enabled_channels = self.get_enabled_channels()
            if not enabled_channels:
                print("No channels enabled, skipping save")
                return
            eeg_data = data[enabled_channels, :]
            
            # Trim to 5 seconds
            samples_per_second = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
            expected_samples = int(self.task_duration * samples_per_second)
            if eeg_data.shape[1] >= expected_samples:
                eeg_data = eeg_data[:, :expected_samples]
                
            # Save as FIF file
            self.save_to_fif(eeg_data)
            
    def save_to_fif(self, eeg_data):
        """Save EEG data as FIF file"""
        # Only use names of enabled channels
        all_channels = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
        enabled_indices = self.get_enabled_channels()
        ch_names = []
        for i, ch_idx in enumerate(enabled_indices):
            if self.channel_states[all_channels[i]]:
                ch_names.append(all_channels[i])
        
        # Verify channel count matches data
        if len(ch_names) != eeg_data.shape[0]:
            raise ValueError(f"Channel count mismatch: {len(ch_names)} names but {eeg_data.shape[0]} channels in data")
        
        sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
        
        # Create info object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * len(ch_names))
        
        # Create raw object
        raw = mne.io.RawArray(eeg_data, info)
        
        # Set montage for proper electrode positions
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Save file
        filename = os.path.join(self.raw_dir, f"session{self.current_session_num}_{int(time.time())}_raw.fif")
        raw.save(filename, overwrite=True)
        print(f"Saved data with channels: {ch_names}")
        return raw

    def _read_annotations_file(self, session_num):
        """Read tasks from annotations file for a specific session"""
        tasks = []
        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()
            in_session = False
            for line in lines:
                line = line.strip()
                if line == f'session{session_num}':
                    in_session = True
                    continue
                elif line.startswith('_') and in_session:
                    break
                elif in_session and line and not line.startswith('session'):
                    tasks.append(line)
        return tasks

    def analyze_session(self, session_num):
        """Analyze recorded session data"""
        # First read the annotations to see what tasks we have
        tasks = self._read_annotations_file(session_num)
        print(f"\nTasks from annotations file for session {session_num}:")
        print(tasks)

        # Load the combined .fif file
        self.raw_data = None  # Store as instance variable
        current_time = 0
        annotations = mne.Annotations(onset=[], duration=[], description=[])

        for raw_file in sorted(os.listdir(self.raw_dir)):
            if f'session{session_num}' in raw_file and raw_file.endswith('_raw.fif'):
                raw = mne.io.read_raw_fif(os.path.join(self.raw_dir, raw_file), preload=True)
                if self.raw_data is None:
                    self.raw_data = raw
                else:
                    self.raw_data.append(raw)
                # Add annotation for this segment
                if len(annotations.onset) < len(tasks):
                    annotations.append(
                        onset=current_time,
                        duration=self.task_duration,
                        description=tasks[len(annotations.onset)]
                    )
                current_time += self.task_duration

        # Set montage for proper plotting
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw_data.set_montage(montage)

        # Set annotations
        self.raw_data.set_annotations(annotations)
        print("\nAdded annotations:")
        print(f"Onsets: {annotations.onset}")
        print(f"Descriptions: {annotations.description}")

        # Get events and epochs
        events, event_id = mne.events_from_annotations(self.raw_data)
        print(f"\nEvents from raw data:")
        print(f"Event IDs: {event_id}")
        print(f"Events shape: {events.shape}")
        print(f"Unique event values: {np.unique(events[:, -1])}")

        epochs = mne.Epochs(self.raw_data, events, event_id=event_id,
                          tmin=0, tmax=self.task_duration,
                          baseline=None, preload=True)

        # Get data and labels
        X = epochs.get_data()
        y = np.zeros(len(epochs.events))
        
        # Map events to labels using exact task names from annotations
        event_mapping = {}
        for i, task in enumerate(sorted(set(tasks))):  # Use actual tasks from file
            event_mapping[task] = i
        print(f"\nEvent mapping from tasks: {event_mapping}")
        
        for i, event in enumerate(epochs.events[:, -1]):
            for name, value in event_id.items():
                if value == event:
                    if name in event_mapping:
                        y[i] = event_mapping[name]
                        print(f"Mapped event {name} to label {event_mapping[name]}")

        # Create CSP patterns
        patterns = []
        class_names = list(event_mapping.keys())  # Use actual task names
        n_components = len(epochs.ch_names)

        for class_idx in range(len(class_names)):
            binary_y = (y == class_idx).astype(int)
            if len(np.unique(binary_y)) < 2:
                print(f"Skipping {class_names[class_idx]} - insufficient data")
                patterns.append(np.zeros((n_components, n_components)))
                continue

            csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
            csp.fit(X, binary_y)
            patterns.append(csp.patterns_)

        return patterns, class_names

    def set_channel_state(self, channel_name, enabled):
        """Enable/disable a channel"""
        self.channel_states[channel_name] = enabled

    def get_enabled_channels(self):
        """Get list of enabled channel indices"""
        all_channels = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)[:6]
        return [ch_idx for i, ch_idx in enumerate(eeg_channels) 
                if self.channel_states[all_channels[i]]]

    def get_channel_positions(self):
        """Get channel positions in 2D for topographic plotting"""
        positions = {
            'FCz': [0, 0.25],
            'C3': [-0.5, 0],
            'Cz': [0, 0],
            'CPz': [0, -0.25],
            'C2': [0.25, 0],
            'C4': [0.5, 0]
        }
        return positions

    # Standard 10-20 and extended 10-20 channel names
    VALID_CHANNELS = {
        # 10-20 standard positions
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 
        'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2',
        # 10-10 extension
        'Fpz', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F9', 'F5', 'F1', 'F2', 'F6', 'F10',
        'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
        'T9', 'T7', 'C5', 'C1', 'C2', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3',
        'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P1', 'P2',
        'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'PO10', 'O9', 'Oz',
        'O10',
        # Additional positions sometimes used
        'A1', 'A2', 'M1', 'M2', 'Nz'
    }
    
    @staticmethod
    def validate_channel_name(name):
        """
        Validate if a channel name follows 10-20 system naming convention
        Returns: (bool, str) - (is_valid, error_message)
        """
        name = name.strip()
        if not name:
            return False, "Channel name cannot be empty"
            
        # Convert to proper case (first letter capital, rest lowercase)
        formatted_name = name[0].upper() + name[1:].lower()
        
        if formatted_name in EEGManager.VALID_CHANNELS:
            return True, ""
            
        # Check if it follows 10-20 naming pattern but isn't in our list
        import re
        pattern = r'^[A-Za-z]+\d*$'
        if not re.match(pattern, name):
            return False, "Invalid channel name format"
            
        return False, f"Channel name '{name}' is not a standard 10-20 system position"

    def get_current_data(self):
        """Get current data for plotting"""
        if self.board and self.board.is_prepared():
            data = self.board.get_current_board_data(250)  # Get 1 second
            if data is not None and data.size > 0:
                enabled_channels = self.get_enabled_channels()
                if not enabled_channels:  # If no channels enabled
                    return None
                return data[enabled_channels, :] * 1e6  # Convert to microvolts
        return None

    def stop_session(self):
        """Stop current session"""
        if self.is_recording:
            self.stop_recording()
            
        # Write session end marker
        with open(self.annotation_file, "a+") as f:
            f.write("_____________________________________________\n")
            
        # Clear session state AFTER all files are saved
        session_num = self.current_session_num  # Store current session number
        self.task_sequence = []
        self.current_task_index = 0
        self.current_session_num = session_num  # Keep it until next session starts

    def combine_fif_files(self):
        """Combine all FIF files into one in formatted_data"""
        print("\n=== Starting File Combination Process ===")
        
        # First, get all raw FIF files for current session
        raw_files = sorted([f for f in os.listdir(self.raw_dir) 
                           if f.endswith('_raw.fif') and 
                           f'session{self.current_session_num}' in f])
        
        print(f"Found {len(raw_files)} raw files:")
        for f in raw_files:
            print(f"  - {f}")
        
        if not raw_files:
            raise ValueError("No FIF files found for current session.")
            
        raw_combined = None
        current_time = 0
        
        # Read annotations from annotations.txt
        with open(self.annotation_file, "r") as f:
            annotation_data = f.read().split("_____________________________________________")
        
        # Initialize lists for annotations
        all_onsets = []
        all_durations = []
        all_descriptions = []
        
        for raw_file in raw_files:
            full_path = os.path.join(self.raw_dir, raw_file)
            print(f"Loading {full_path}")
            raw = mne.io.read_raw_fif(full_path, preload=True)
            
            if raw_combined is None:
                raw_combined = raw
            else:
                raw_combined.append(raw)
            
            # Update current time
            current_time += len(raw.times) / raw.info['sfreq']
        
        # Get tasks for current session
        tasks = self._read_annotations_file(self.current_session_num)
        
        # Create annotations
        for i, task in enumerate(tasks):
            all_onsets.append(i * self.task_duration)
            all_durations.append(self.task_duration)
            task = task.strip().lower()
            if task == "jaw_clenching":
                task = "jaw"
            elif task.endswith("_hand"):
                task = task.split("_")[0]
            all_descriptions.append(task)
        
        # Set annotations
        annotations = mne.Annotations(
            onset=all_onsets,
            duration=all_durations,
            description=all_descriptions
        )
        raw_combined.set_annotations(annotations)
        
        # Save combined file
        output_file = os.path.join(self.formatted_dir, 'S03.fif')
        raw_combined.save(output_file, overwrite=True)
        print(f"Saved combined file to {output_file}")
        
        return output_file