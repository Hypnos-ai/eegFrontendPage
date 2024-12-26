import numpy as np
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import os
import time
import random
from mne.decoding import CSP
import requests
import base64

class EEGManager:
    def __init__(self):
        # Board setup
        self.board = None
        self.is_recording = False
        self.is_streaming = False
        
        # Session management
        self.current_session_num = None
        self.next_session_number = self._get_next_session_number()
        print(os.path.dirname(__file__))

        # File management
        base_dir = "C:\\NeuroSync"
        self.data_dir = os.path.join(base_dir, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.formatted_dir = os.path.join(self.data_dir, 'formatted')
        self.annotation_file = os.path.join(self.data_dir, 'annotations.txt')
        self.sample_dir = os.path.join(base_dir, 'sample_data')
        
        # Ensure the formatted directory exists
        self.formatted_sample_dir = os.path.join(self.sample_dir, 'formatted')
        
        # Create directories and annotation file if they don't exist
        for directory in [self.data_dir, self.raw_dir, self.processed_dir, self.formatted_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create empty annotation file if it doesn't exist
        if not os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'w') as f:
                f.write('')
            
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
        
        # Task setup with default tasks
        self.task_names = {
            "Right_Hand": ["Imagine or perform right hand movement", 21],
            "Left_Hand": ["Imagine or perform left hand movement", 21],
            "Blinking": ["Blink your eyes at a comfortable pace", 21],
            "Jaw_Clenching": ["Clench your jaw with moderate force", 21],
            "Relax": ["Relax your body and mind, do nothing", 21]
        }
        self.task_sequence = []
        self.current_task_index = 0
        self.break_duration = 3  # Default break duration
        
        # Initialize board
        self.initialize_board()
        
        # Initialize session counter
        self.next_session_number = self._get_next_session_number()
        
        # Data storage
        self.raw_data = None
        
        # API setup
        self.api_url = "your-vercel-deployment-url/api"
        
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
        self.current_session_num = session_num
        self.task_sequence = []
        self.current_task_index = 0
        
        # Create balanced task sequence from current task_names
        for task_name, (instruction, count) in self.task_names.items():
            self.task_sequence.extend([(task_name, instruction)] * count)
            
        # Randomize task order
        random.shuffle(self.task_sequence)
        
        # Write session header to annotations file
        with open(self.annotation_file, "a+") as f:
            f.write(f"\nsession{self.current_session_num}\n")
        
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
        try:
            if not os.path.exists(self.annotation_file):
                return []
            
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            
            tasks = []
            in_session = False
            for line in lines:
                line = line.strip()
                if line == f'session{session_num}':
                    in_session = True
                elif line.startswith('_'):
                    in_session = False
                elif in_session and line:
                    tasks.append(line)
                
            return tasks
        except Exception as e:
            print(f"Error reading annotations: {e}")
            return []

    def analyze_session(self, session_num):
        """Analyze recorded session data"""
        try:
            # Handle tasks differently for sample data
            if session_num == 0:  # Sample data
                # Read sample annotations directly from sample_data folder
                sample_annotations = os.path.join(self.sample_dir, 'annotations.txt')
                if not os.path.exists(sample_annotations):
                    raise ValueError("Sample annotations file not found")
                
                # Parse sample annotations file
                with open(sample_annotations, 'r') as f:
                    lines = f.readlines()
                    tasks = []
                    in_session = False
                    for line in lines:
                        line = line.strip()
                        if line == 'session1':  # First session in sample data
                            in_session = True
                        elif line.startswith('_'):
                            break  # Stop at first session boundary
                        elif in_session and line:
                            # Clean up task names
                            task = line.strip().lower()
                            if task == "jaw_clenching":
                                task = "jaw"
                            elif task.endswith("_hand"):
                                task = task.split("_")[0]
                            tasks.append(task)
            else:
                # Regular session - read from main annotations file
                tasks = self._read_annotations_file(session_num)
            
            if not tasks:
                raise ValueError("No tasks found for this session")
            print(f"\nTasks from annotations file for session {session_num}:")
            print(tasks)

            # Load the appropriate data file
            if session_num == 0:  # Sample data
                sample_file = os.path.join(self.sample_dir, 'formatted', 'S03.fif')
                if not os.path.exists(sample_file):
                    raise ValueError("Sample data file (S03.fif) not found")
                self.raw_data = mne.io.read_raw_fif(sample_file, preload=True)
            else:
                # Regular session handling remains the same
                combined_file = os.path.join(self.formatted_dir, f'session{session_num}_combined.fif')
                if os.path.exists(combined_file):
                    self.raw_data = mne.io.read_raw_fif(combined_file, preload=True)
                else:
                    # Get all raw files for this session
                    raw_files = sorted([f for f in os.listdir(self.raw_dir) 
                                      if f'session{session_num}' in f and f.endswith('_raw.fif')])
                    if not raw_files:
                        raise ValueError("No recording files found for this session")

                    # Load and combine all files
                    raw_combined = None
                    for raw_file in raw_files:
                        raw = mne.io.read_raw_fif(os.path.join(self.raw_dir, raw_file), preload=True)
                        if raw_combined is None:
                            raw_combined = raw
                        else:
                            raw_combined.append(raw)
                    self.raw_data = raw_combined

            # Create epochs
            events = mne.events_from_annotations(self.raw_data)
            epochs = mne.Epochs(self.raw_data, events[0], event_id=events[1],
                              tmin=0.0, tmax=5.0, baseline=None, preload=True)

            # Get data ready for CSP
            X = epochs.get_data()
            
            # Create event mapping using cleaned task names
            unique_events = sorted(list(set(tasks)))  # Sort to ensure consistent ordering
            event_mapping = {event: idx for idx, event in enumerate(unique_events)}
            event_id = {task: i+1 for i, task in enumerate(unique_events)}
            
            print("Event mapping:", event_mapping)
            print("Event IDs:", event_id)
            
            # Initialize labels array
            y = np.zeros(len(epochs.events))
            
            # Map events to labels
            for i, event in enumerate(epochs.events[:, -1]):
                event_name = None
                for name, value in event_id.items():
                    if value == event:
                        event_name = name
                        break
                if event_name and event_name in event_mapping:
                    y[i] = event_mapping[event_name]

            # Create CSP patterns
            patterns = []
            class_names = unique_events
            n_components = len(epochs.ch_names)

            for class_idx in range(len(class_names)):
                binary_y = (y == class_idx).astype(int)
                if sum(binary_y) < 2:
                    print(f"Skipping {class_names[class_idx]} - insufficient data")
                    patterns.append(np.zeros((n_components, n_components)))
                    continue

                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
                csp.fit(X, binary_y)
                patterns.append(csp.patterns_)

            return patterns, class_names
            
        except Exception as e:
            print(f"Analysis error details: {str(e)}")
            raise

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
                           if f'session{self.current_session_num}' in f and f.endswith('_raw.fif')])
        
        print(f"Found {len(raw_files)} raw files:")
        for f in raw_files:
            print(f"  - {f}")
        
        if not raw_files:
            raise ValueError("No FIF files found for current session.")
            
        raw_combined = None
        current_time = 0
        
        # Get tasks for current session
        tasks = self._read_annotations_file(self.current_session_num)
        
        # Load first file to get recording duration
        first_raw = mne.io.read_raw_fif(os.path.join(self.raw_dir, raw_files[0]), preload=True)
        recording_duration = len(first_raw.times) / first_raw.info['sfreq']
        
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
        
        # Create annotations using actual recording duration
        for i, task in enumerate(tasks):
            all_onsets.append(i * recording_duration)
            all_durations.append(recording_duration)
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

    def update_action_names(self, new_actions):
        """Update the action names and their configurations"""
        self.task_names = new_actions

    def delete_session(self, session_num):
        """Delete a specific session and its files"""
        # Delete raw files
        for file in os.listdir(self.raw_dir):
            if f'session{session_num}_' in file:
                os.remove(os.path.join(self.raw_dir, file))
            
        # Delete processed files
        for file in os.listdir(self.processed_dir):
            if f'session{session_num}' in file:
                os.remove(os.path.join(self.processed_dir, file))
            
        # Delete formatted files
        for file in os.listdir(self.formatted_dir):
            if f'session{session_num}' in file:
                os.remove(os.path.join(self.formatted_dir, file))
            
        # Update annotations file
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            
            # Write back all lines except those for the deleted session
            with open(self.annotation_file, 'w') as f:
                in_session = False
                for line in lines:
                    if f'session{session_num}' in line:
                        in_session = True
                    elif line.startswith('_'):
                        in_session = False
                    elif not in_session:
                        f.write(line)

    def delete_all_sessions(self):
        """Delete all session files and reset"""
        # Delete all files in data directories
        for directory in [self.raw_dir, self.processed_dir, self.formatted_dir]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        # Clear annotations file
        with open(self.annotation_file, 'w') as f:
            f.write('')
        
        # Reset session counter
        self.next_session_number = 1
        self.current_session_num = None

    def combine_all_fif_files(self):
        """Combine FIF files from all sessions into one"""
        print("\n=== Starting All Sessions File Combination Process ===")
        
        # Get all raw FIF files
        raw_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('_raw.fif')])
        
        print(f"Found {len(raw_files)} total raw files:")
        for f in raw_files:
            print(f"  - {f}")
        
        if not raw_files:
            raise ValueError("No FIF files found.")
        
        raw_combined = None
        current_time = 0
        
        # Initialize lists for annotations
        all_onsets = []
        all_durations = []
        all_descriptions = []
        
        # Get all session numbers
        sessions = sorted(list(set([
            int(f.split('_')[0].replace('session', '')) 
            for f in raw_files
        ])))
        
        # Process each session
        for session in sessions:
            # Get tasks for this session
            tasks = self._read_annotations_file(session)
            
            # Add tasks to annotations
            for i, task in enumerate(tasks):
                all_onsets.append(current_time)
                all_durations.append(self.task_duration)
                task = task.strip().lower()
                if task == "jaw_clenching":
                    task = "jaw"
                elif task.endswith("_hand"):
                    task = task.split("_")[0]
                all_descriptions.append(task)
                current_time += self.task_duration + self.break_duration
        
        # Combine all files
        for raw_file in raw_files:
            full_path = os.path.join(self.raw_dir, raw_file)
            print(f"Loading {full_path}")
            raw = mne.io.read_raw_fif(full_path, preload=True)
            
            if raw_combined is None:
                raw_combined = raw
            else:
                raw_combined.append(raw)
        
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

    def save_session_data(self, session_num, data):
        """Save session data to remote storage"""
        try:
            response = requests.post(
                f"{self.api_url}/session_data",
                json={
                    'session_num': session_num,
                    'session_data': base64.b64encode(data).decode()
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error saving session data: {e}")
            return False
    
    def load_session_data(self, session_num):
        """Load session data from remote storage"""
        try:
            response = requests.get(f"{self.api_url}/session_data/{session_num}")
            response.raise_for_status()
            data = response.json()
            return base64.b64decode(data['session_data'])
        except Exception as e:
            print(f"Error loading session data: {e}")
            return None

    def get_sample_files(self):
        """Get list of available sample data files"""
        try:
            # Look specifically for S03.fif in the formatted subfolder of sample_data
            formatted_dir = os.path.join(self.sample_dir, 'formatted')
            sample_file = os.path.join(formatted_dir, 'S03.fif')
            
            if os.path.exists(sample_file):
                return [sample_file]
            else:
                print(f"Sample file not found at: {sample_file}")
                return []
            
        except Exception as e:
            print(f"Error getting sample files: {e}")
            return []

    def update_channels(self, channel_names):
        """Update the active channel configuration"""
        try:
            # Update channel names
            self.channel_names = channel_names
            
            # Update channel states
            self.channel_states = {name: True for name in channel_names}
            
            # If board is connected, update channel configuration
            if self.board and self.board.is_prepared():
                # Update board channel settings if needed
                pass
            
            return True
            
        except Exception as e:
            print(f"Error updating channels: {str(e)}")
            return False