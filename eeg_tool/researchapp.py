# app.py
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import mne
from AdaptiveDQN_RLEEGNET import AdaptiveDQNRLEEGNET
import json
import os
import time
import atexit
from config import Config
from flask_cors import CORS
import random
from mne.decoding import CSP

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class EEGManager:
    def __init__(self):
        self.board = None
        self.is_recording = False
        self.current_session = None
        self.current_trial = None
        self.dqn_model = AdaptiveDQNRLEEGNET()
        
        # Add task-related attributes
        self.task_names = {
            "Right_Hand": "Imagine or perform right hand movement",
            "Left_Hand": "Imagine or perform left hand movement",
            "Blinking": "Blink your eyes at a comfortable pace",
            "Jaw_Clenching": "Clench your jaw with moderate force",
            "Relax": "Relax your body and mind, do nothing"
        }
        self.task_sequence = []
        self.current_task_index = 0
        self.task_duration = 5  # Fixed at 5 seconds per trial
        self.break_duration = 3  # Default, can be changed
        self.samples_per_action = 21  # Default, can be changed
        self.session_start_time = 30
        
        # File paths
        self.data_dir = 'collected_data'
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.formatted_dir = 'formatted_data'
        self.annotation_file = os.path.join(self.data_dir, 'annotations.txt')
        
        print("\n=== Initializing EEG Manager ===")
        print(f"Working directory: {os.getcwd()}")
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.raw_dir, self.processed_dir, self.formatted_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory: {directory}")
            print(f"  Exists: {os.path.exists(directory)}")
            print(f"  Writable: {os.access(directory, os.W_OK)}")
        
        # Create empty annotations file if it doesn't exist
        if not os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'w') as f:
                pass
            print(f"Created annotations file: {self.annotation_file}")
        
        # File management
        self.current_session_num = None
        self.annotations = []
        
        # Current recording data
        self.current_eeg_data = []
        self.current_timestamps = []
        
        # Print board info
        print("\n=== Board Configuration ===")
        print(f"Board ID: {BoardIds.CYTON_BOARD.value}")
        print(f"EEG channels: {BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)}")
        print(f"Sampling rate: {BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)}")
        print(f"Timestamp channel: {BoardShim.get_timestamp_channel(BoardIds.CYTON_BOARD.value)}")
        print(f"Package size: {BoardShim.get_package_num_channel(BoardIds.CYTON_BOARD.value)}")
    
    def initialize_board(self, port='COM9'):
        """Initialize the EEG board with proper error handling"""
        if not port:
            raise ValueError("Port cannot be empty")
        
        try:
            if self.board and self.board.is_prepared():
                try:
                    self.stop_stream()
                    self.board.release_session()
                except Exception as e:
                    print(f"Warning: Error releasing previous session: {e}")
                
            params = BrainFlowInputParams()
            params.serial_port = port
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            self.board.prepare_session()
            print("Board prepared, testing data acquisition...")
            
            # Test data acquisition
            self.board.start_stream()
            print("Stream started")
            time.sleep(1)  # Wait for data
            print("Getting test data...")
            data = self.board.get_current_board_data(250)
            print(f"Test data shape: {data.shape if data is not None else None}")
            
            if data is None or data.size == 0:
                self.board.stop_stream()
                raise Exception("Board initialized but not receiving data")
            
            print(f"Test data received, shape: {data.shape}")
            print(f"Board initialized successfully on port {port}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "BOARD_NOT_READY_ERROR" in error_msg:
                error_msg = "Board not ready. Please check if the device is powered on and properly connected."
            elif "PORT_ALREADY_OPEN_ERROR" in error_msg:
                error_msg = "Port already in use. Please close other applications using this port."
            print(f"Error initializing board: {error_msg}")
            self.board = None
            raise Exception(error_msg)
        
    def start_stream(self):
        if self.board:
            self.board.start_stream()
            
    def stop_stream(self):
        if self.board:
            self.board.stop_stream()
            
    def get_current_data(self):
        """Get current data without storing"""
        if self.board and self.board.is_prepared():
            try:
                data = self.board.get_current_board_data(250)
                if data is not None and data.size > 0 and data.shape[1] > 0:
                    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
                    return data[eeg_channels[:6], :]
            except Exception as e:
                print(f"Error getting data: {e}")
        return None
        
    def start_recording(self):
        """Start recording EEG data"""
        print("\n=== Starting Recording ===")
        # Clear any existing data
        self.current_eeg_data = []
        self.current_timestamps = []
        self.is_recording = True
        
        # Start fresh data collection
        if self.board and self.board.is_prepared():
            # Clear board's data buffer before starting new recording
            self.board.get_board_data()  # This clears the buffer
            print("Cleared board buffer")

        # Record the current task if we're in a session
        if self.current_task_index > 0 and self.current_task_index <= len(self.task_sequence):
            current_task = self.task_sequence[self.current_task_index - 1]
            task_name = current_task[0]  # Get task name from tuple
            self.record_task(task_name)
            print(f"Recording task: {task_name}")
            print(f"Session number: {self.current_session_num}")

    def stop_recording(self):
        """Stop recording and save data"""
        if not self.is_recording:
            return
            
        print(f"\n=== Stopping Recording ===")
        
        self.is_recording = False
        
        if self.board and self.board.is_prepared():
            try:
                # Get all data since last recording start
                data = self.board.get_board_data()
                eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
                eeg_data = data[eeg_channels[:6], :]  # Get first 6 EEG channels
                
                print(f"Collected data shape: {eeg_data.shape}")
                sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
                expected_samples = int(self.task_duration * sampling_rate)
                print(f"Expected samples for {self.task_duration}s: {expected_samples}")

                # Ensure we have the right amount of data (5 seconds worth)
                if eeg_data.shape[1] >= expected_samples:
                    eeg_data = eeg_data[:, :expected_samples]  # Take exactly 5 seconds
                    print(f"Trimmed data to {self.task_duration}s: {eeg_data.shape}")
                
                if self.current_session_num:
                    try:
                        # Save as FIF file
                        raw_file = os.path.join(self.raw_dir, f"session{self.current_session_num}_raw.fif")
                        
                        # Create info for raw object
                        ch_names = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
                        info = mne.create_info(
                            ch_names=ch_names,
                            sfreq=sampling_rate,
                            ch_types=['eeg'] * len(ch_names)
                        )
                        
                        # Create raw object and save
                        raw = mne.io.RawArray(eeg_data, info)
                        
                        # Get current task name
                        if self.current_task_index > 0 and self.current_task_index <= len(self.task_sequence):
                            current_task = self.task_sequence[self.current_task_index - 1]
                            task_name = current_task[0].split('_')[0].lower()
                            
                            # Add annotation for this task
                            onset = 0  # Start of the data
                            duration = self.task_duration  # 5 seconds
                            description = task_name
                            annot = mne.Annotations(onset=[onset], 
                                                  duration=[duration],
                                                  description=[description])
                            raw.set_annotations(annot)
                            print(f"Added annotation: {description} at {onset}s for {duration}s")
                        
                        raw.save(raw_file, overwrite=True)
                        print(f"Saved raw data to {raw_file}")
                        print(f"File exists: {os.path.exists(raw_file)}")
                        print(f"File size: {os.path.getsize(raw_file)} bytes")
                        
                    except Exception as e:
                        print(f"Error saving recording: {e}")
                        import traceback
                        print(traceback.format_exc())
            except Exception as e:
                print(f"Error getting board data: {e}")
                import traceback
                print(traceback.format_exc())

    def save_to_bdf(self, filename):
        """Save current recording to BDF file"""
        if not self.current_eeg_data:
            raise ValueError("No data to save")
            
        print(f"\nSaving data to {filename}")
        print(f"Number of data chunks: {len(self.current_eeg_data)}")
        
        try:
            # Convert list of arrays to one continuous array
            eeg_data = np.concatenate(self.current_eeg_data, axis=1)
            print(f"Data shape after concatenation: {eeg_data.shape}")
            
            # Create info for raw object
            ch_names = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=250.0,
                ch_types=['eeg'] * len(ch_names)
            )
            
            # Create raw object and save as BDF
            raw = mne.io.RawArray(eeg_data, info)
            raw.save(filename.replace('.bdf', '.fif'), overwrite=True)
            print(f"Successfully saved data to {filename}")
            print(f"Data duration: {raw.times[-1]:.2f} seconds")
            
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False

    def train_model(self):
        try:
            # Load saved dataset
            dataset = np.load('eeg_dataset.npy')
            
            # Process data through CSP and feature extraction
            processed_data = []
            for trial in dataset:
                features = self.dqn_model.featuresarray_load(data_array=trial)
                processed_data.append(features)
            
            processed_data = np.array(processed_data)
            
            # Create environment with processed data
            Plasticity = self.dqn_model.create_environment()
            env = Plasticity(dataset=(processed_data, np.zeros(len(processed_data))))  # Add proper labels
            
            # Setup and train model
            model, callbacks = self.dqn_model.setup_training(env)
            model.learn(total_timesteps=2500, callback=callbacks)
            
            # Save the trained model
            model.save("dqn_plasticity_final")
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def initialize_session(self, session_num, samples_per_action=21, break_duration=3):
        """Initialize a new recording session with randomized tasks"""
        # Get next available session number if current one exists
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, "r") as f:
                content = f.read()
                existing_sessions = [int(s.split('session')[1]) 
                                   for s in content.split('\n') 
                                   if s.startswith('session')]
                if existing_sessions:
                    next_session = max(existing_sessions) + 1
                    session_num = next_session
        
        self.current_session_num = session_num
        self.samples_per_action = samples_per_action
        self.break_duration = break_duration
        self.task_sequence = []
        self.current_task_index = 0
        
        # Create balanced task sequence
        for task_name, instruction in self.task_names.items():
            self.task_sequence.extend([(task_name, instruction)] * self.samples_per_action)
        
        # Randomize task order
        random.shuffle(self.task_sequence)
        
        # Initialize annotations file with proper session marker
        with open(self.annotation_file, "a+") as f:
            # Add newline if file is not empty
            f.seek(0, 2)  # Go to end of file
            if f.tell() > 0:  # If file is not empty
                f.write("\n")
            f.write(f"session{session_num}\n")
            
        return {
            'total_tasks': len(self.task_sequence),
            'session_num': session_num
        }
        
    def get_next_task(self):
        """Get the next task in the sequence"""
        if self.current_task_index < len(self.task_sequence):
            task = self.task_sequence[self.current_task_index]
            self.current_task_index += 1
            return task
        return None
        
    def record_task(self, task_name):
        """Record task annotation"""
        with open(self.annotation_file, "a+") as f:
            f.write(f"{task_name.split('_')[0]}\n")
            
    def save_as_annotated_fif(self):
        """Convert BDF to annotated FIF file"""
        import mne
        
        # Define task label mapping
        task_labels = {
            "Relax": "relax",
            "Right": "right_hand",
            "Left": "left_hand",
            "Blinking": "blinking",
            "Jaw": "jaw_clenching"
        }
        
        # Read BDF file
        bdf_file = os.path.join(self.raw_dir, f"session{self.current_session_num}.bdf")
        if not os.path.exists(bdf_file):
            raise FileNotFoundError(f"BDF file not found: {bdf_file}")
            
        raw = mne.io.read_raw_bdf(bdf_file, preload=True)
        raw.drop_channels(['EEG 7', 'EEG 8', 'Accel X', 'Accel Y', 'Accel Z'])
        
        # Read annotations
        with open(self.annotation_file, "r") as f:
            data = f.read().split("_____________________________________________")
        
        # Find current session
        session_index = next(i for i, section in enumerate(data) 
                           if f"session{self.current_session_num}" in section)
        session_data = data[session_index].strip().split("\n")[1:]
        
        # Create annotations
        onsets = []
        durations = []
        descriptions = []
        
        current_time = self.session_start_time
        for task_name in session_data:
            if task_name in task_labels:
                descriptions.append(task_labels[task_name])
                onsets.append(current_time)
                durations.append(self.task_duration)
                current_time += self.task_duration + self.break_duration
                
        # Set annotations
        annotations = mne.Annotations(onset=onsets, duration=durations, 
                                   description=descriptions)
        raw.set_annotations(annotations)
        
        # Save annotated file
        annotated_file = os.path.join(self.processed_dir, 
            f"session{self.current_session_num}_annotated.fif")
        raw.save(annotated_file, overwrite=True)
        
        return annotated_file

    def combine_fif_files(self):
        """Combine all FIF files into one in formatted_data"""
        import mne
        
        print("\n=== Starting File Combination Process ===")
        
        # First, get all raw FIF files
        raw_files = sorted([f for f in os.listdir(self.raw_dir) 
                           if f.endswith('_raw.fif')])
        
        print(f"Found {len(raw_files)} raw files:")
        for f in raw_files:
            print(f"  - {f}")
        
        if not raw_files:
            raise ValueError("No FIF files found in raw directory.")
            
        raw_combined = None
        current_time = 0  # Keep track of time for annotations
        
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
            raw = mne.io.read_raw_fif(
                full_path,
                preload=True
            )
            print(f"Loaded {raw_file}")
            
            # Get session number from filename
            session_num = int(raw_file.split('session')[1].split('_')[0])
            
            # Find corresponding annotations
            session_annotations = next((section for section in annotation_data 
                                    if f"session{session_num}" in section), None)
            if not session_annotations:
                print(f"Warning: No annotations found for session {session_num}")
                continue
            
            # Skip the session header line and empty lines
            tasks = [task.strip() for task in session_annotations.split('\n')[1:] 
                    if task.strip() and not task.startswith('session')]
            
            # Add annotations for this file
            for i, task in enumerate(tasks):
                all_onsets.append(current_time + (i * (self.task_duration + self.break_duration)))
                all_durations.append(self.task_duration)
                # Clean up task name and ensure it matches our expected format
                task = task.strip().lower()
                if task == "jaw_clenching":
                    task = "jaw"
                elif task.endswith("_hand"):
                    task = task.split("_")[0]  # Convert "left_hand" to "left"
                all_descriptions.append(task)
            
            if raw_combined is None:
                raw_combined = raw
                print("Set as base for combined file")
            else:
                raw_combined.append(raw)
                print(f"Appended to combined file. Total length: {len(raw_combined.times)} samples")
            
            # Update current time
            current_time += len(raw.times) / raw.info['sfreq']
        
        # Create and set annotations
        annotations = mne.Annotations(onset=all_onsets,
                                    duration=all_durations,
                                    description=all_descriptions)
        raw_combined.set_annotations(annotations)
        print("\nAdded annotations:")
        print(f"Total annotations: {len(annotations)}")
        print(f"Unique tasks: {set(all_descriptions)}")
        
        # Save combined file
        combined_file = os.path.join(self.formatted_dir, 'S03.fif')
        raw_combined.save(combined_file, overwrite=True)
        print(f"\nSaved combined file:")
        print(f"  Path: {combined_file}")
        print(f"  Size: {os.path.getsize(combined_file)} bytes")
        print("=== File Combination Complete ===\n")
        
        return combined_file

eeg_manager = EEGManager()

@app.route('/')
def index():
    return render_template('researchindex.html')

@app.route('/initialize', methods=['POST'])
def initialize_board():
    try:
        # Ensure request has proper content type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 400

        # Get port from request, default to Config.BOARD_PORT if not provided
        data = request.get_json()
        port = data.get('port', Config.BOARD_PORT) if data else Config.BOARD_PORT
        
        # Initialize board
        success = eeg_manager.initialize_board(port)
        if success:
            socketio.emit('board_status', {
                'status': 'success',
                'message': f'Board initialized on port {port}'
            })
            return jsonify({
                'status': 'success',
                'message': f'Board initialized on port {port}'
            })
        else:
            error_msg = f'Failed to initialize board on port {port}'
            socketio.emit('board_status', {
                'status': 'error',
                'message': error_msg
            })
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400
            
    except Exception as e:
        error_msg = str(e)
        socketio.emit('board_status', {
            'status': 'error',
            'message': error_msg
        })
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        duration = request.json.get('duration', 5)
        task_name = request.json.get('task_name')  # Get task name from request
        eeg_manager.start_recording()
        socketio.emit('recording_status', {
            'status': 'started',
            'duration': duration,
            'timestamp': time.time()
        })
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        eeg_manager.stop_recording()
        socketio.emit('recording_status', {
            'status': 'stopped',
            'timestamp': time.time()
        })
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        success = eeg_manager.train_model()
        if success:
            socketio.emit('training_status', {
                'status': 'success',
                'message': 'Model training completed successfully'
            })
            return jsonify({'status': 'success'})
        else:
            socketio.emit('training_status', {
                'status': 'error',
                'message': 'Model training failed'
            })
            return jsonify({'status': 'error'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    # Send initial connection status
    socketio.emit('connection_status', {'status': 'Connected to server'})

def send_eeg_data():
    """Background task to continuously send EEG data to clients"""
    print("\n=== Starting EEG Data Stream ===")
    while True:
        if eeg_manager.board and eeg_manager.board.is_prepared():
            try:
                data = eeg_manager.board.get_current_board_data(250)  # Get 1 second of data
                if data is not None and data.size > 0 and data.shape[1] > 0:
                    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
                    eeg_data = data[eeg_channels[:6], :]
                    
                    # Scale data to microvolts
                    eeg_data = eeg_data * 1e6  # Convert to microvolts
                    
                    # Send data to frontend
                    socketio.emit('eeg_data', {
                        'data': eeg_data.tolist(),
                        'timestamp': time.time()
                    })
            except Exception as e:
                print(f"Error in data stream: {e}")
                import traceback
                print(traceback.format_exc())
        socketio.sleep(0.02)  # 50Hz update rate for smoother display

def cleanup():
    print("Cleaning up resources...")
    if eeg_manager.board:
        try:
            eeg_manager.stop_stream()
            eeg_manager.board.release_session()
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup)

@socketio.on_error()
def error_handler(e):
    print(f"SocketIO error: {e}")
    socketio.emit('error', {'message': str(e)})

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO default error: {e}")
    socketio.emit('error', {'message': str(e)})

@app.route('/start_session', methods=['POST'])
def start_session():
    try:
        session_num = request.json.get('session_num')
        samples_per_action = request.json.get('samples_per_action', 21)
        break_duration = request.json.get('break_duration', 3)
        
        if not session_num:
            return jsonify({
                'status': 'error',
                'message': 'Session number required'
            }), 400
            
        session_info = eeg_manager.initialize_session(
            session_num, 
            samples_per_action=samples_per_action,
            break_duration=break_duration
        )
        
        return jsonify({
            'status': 'success',
            'total_tasks': session_info['total_tasks'],
            'session_num': session_info['session_num'],
            'task_duration': eeg_manager.task_duration,  # Always 5 seconds
            'break_duration': eeg_manager.break_duration
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/next_task', methods=['GET'])
def get_next_task():
    try:
        task = eeg_manager.get_next_task()
        if task:
            task_name, instruction = task
            return jsonify({
                'status': 'success',
                'task_name': task_name,
                'instruction': instruction,
                'duration': eeg_manager.task_duration,
                'break_duration': eeg_manager.break_duration
            })
        else:
            return jsonify({
                'status': 'complete',
                'message': 'All tasks completed'
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    try:
        # Write session end marker
        with open(eeg_manager.annotation_file, "a+") as f:
            f.write('_____________________________________________\n')

        
            
        # Reset session state
        eeg_manager.current_session_num = None
        eeg_manager.task_sequence = []
        eeg_manager.current_task_index = 0
        
        return jsonify({
            'status': 'success',
            'message': 'Session stopped successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/complete_session', methods=['POST'])
def complete_session():
    try:
        # Save current session as FIF
        annotated_file = eeg_manager.save_as_annotated_fif()
        print(f"Saved annotated file: {annotated_file}")
        
        # Try to combine all FIF files
        try:
            combined_file = eeg_manager.combine_fif_files()
            print(f"Created combined file: {combined_file}")
        except Exception as e:
            print(f"Warning: Could not combine FIF files: {e}")
        
        return jsonify({
            'status': 'success',
            'message': 'Session completed and files processed'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/analyze_session', methods=['POST'])
def analyze_session():
    try:
        # First, ensure we have a combined file
        try:
            combined_file = eeg_manager.combine_fif_files()
            print(f"Created/Updated combined file at: {combined_file}")
        except Exception as e:
            print(f"Error creating combined file: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            })

        # Load the combined .fif file
        raw = mne.io.read_raw_fif(combined_file, preload=True)
        print(raw.info)
        print(f"Available channels: {raw.ch_names}")  # Debug print
        print(f"Annotations: {raw.annotations}")
 
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
 
        # Extract events and event IDs
        events, event_id = mne.events_from_annotations(raw)
        annotation_counts = mne.annotations.count_annotations(raw.annotations)
        print(f"Annotation counts: {annotation_counts}")
        print(f"Found events: {event_id}")  # Debug print
 
        # Create epochs
        tmin, tmax = 0, 5
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=None, preload=True
        )
 
        # Get data and labels
        X = epochs.get_data()
        y = np.zeros(len(epochs.events))
        print("Event mapping:", event_id)  # Debug print
        print("Events shape:", epochs.events.shape)
        print("Unique event values:", np.unique(epochs.events[:, -1]))

        # Map event names to numeric labels
        event_mapping = {
            'left': 0,
            'right': 1,
            'blinking': 2,
            'jaw': 3,
            'relax': 4
        }

        for i, event in enumerate(epochs.events[:, -1]):
            # Get the event name from event_id
            event_name = None
            for name, value in event_id.items():
                if value == event:
                    event_name = name.lower()
                    break
            
            if event_name in event_mapping:
                y[i] = event_mapping[event_name]
            else:
                print(f"Warning: Unknown event {event_name}")

        # Verify we have valid labels
        unique_labels = np.unique(y)
        print("Unique labels:", unique_labels)
        if len(unique_labels) < 2:
            raise ValueError(f"Not enough unique classes in data. Found classes: {unique_labels}")

        # Create CSP for each class
        class_names = ['Left Hand', 'Right Hand', 'Blinking', 'Jaw Clenching', 'Relax']
        n_components = 6
 
        # Store CSP patterns for each class
        csp_patterns = []
        class_counts = []
 
        for class_idx in range(5):
            binary_y = (y == class_idx).astype(int)
            csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
            csp.fit(X, binary_y)
             
            # Store patterns for this class
            patterns = []
            for comp_idx in range(3):  # Store top 3 components
                pattern = csp.patterns_[:, comp_idx].tolist()
                patterns.append(pattern)
            csp_patterns.append(patterns)
             
            # Store class count
            class_counts.append(int(np.sum(y == class_idx)))
 
        return jsonify({
            'status': 'success',
            'csp_patterns': csp_patterns,
            'class_names': class_names,
            'class_counts': class_counts,
            'channel_names': ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4'],
            'event_ids': event_id
        })
 
    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    try:
        socketio.start_background_task(send_eeg_data)
        socketio.run(app, debug=True)
    finally:
        cleanup()