# app.py
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from AdaptiveDQN_RLEEGNET import AdaptiveDQNRLEEGNET
import json
import os
import time

app = Flask(__name__)
socketio = SocketIO(app)

class EEGManager:
    def __init__(self):
        self.board = None
        self.is_recording = False
        self.current_session = None
        self.current_trial = None
        self.dqn_model = AdaptiveDQNRLEEGNET()
        self.annotations = []
        self.session_start_time = 30
        self.task_duration = 5
        self.break_duration = 3
        
    
    def initialize_board(self, port='COM9'):
        try:
            params = BrainFlowInputParams()
            params.serial_port = port
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            self.board.prepare_session()
            return True
        except Exception as e:
            print(f"Error initializing board: {e}")
            return False
        
    def start_stream(self):
        if self.board:
            self.board.start_stream()
            
    def stop_stream(self):
        if self.board:
            self.board.stop_stream()
            
    def get_current_data(self):
        if self.board and self.board.is_prepared():
            try:
                # Get data from board
                data = self.board.get_current_board_data(250)  # 1 second at 250Hz
                if data is not None and data.size > 0:
                    # Extract EEG channels
                    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
                    eeg_data = data[eeg_channels[:6], :]  # Get first 6 EEG channels
                    
                    # Apply basic filtering if needed
                    # eeg_data = self.board.get_current_board_data(250)
                    return eeg_data
            except Exception as e:
                print(f"Error getting data: {e}")
        return None
        
    def start_recording(self):
        self.is_recording = True
        self.current_trial_data = []
        
    def stop_recording(self):
        self.is_recording = False
        if self.current_trial_data:
            self.dataset.append(self.current_trial_data)
            self.save_dataset()
            
    def save_dataset(self):
        np.save('eeg_dataset.npy', np.array(self.dataset))

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

eeg_manager = EEGManager()

@app.route('/')
def index():
    return render_template('researchindex.html')

@app.route('/initialize', methods=['POST'])
def initialize_board():
    port = request.json.get('port', 'COM9')
    try:
        eeg_manager.initialize_board(port)
        eeg_manager.start_stream()
        socketio.emit('board_status', {
            'status': 'success',
            'message': f'Board initialized on port {port}'
        })
        return jsonify({'status': 'success'})
    except Exception as e:
        socketio.emit('board_status', {
            'status': 'error',
            'message': str(e)
        })
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        duration = request.json.get('duration', 5)
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
    while True:
        if eeg_manager.board and eeg_manager.board.is_prepared():
            try:
                data = eeg_manager.get_current_data()
                if data is not None and data.size > 0:
                    # Send raw EEG data
                    socketio.emit('eeg_data', {
                        'data': data.tolist(),
                        'timestamp': time.time(),
                        'is_recording': eeg_manager.is_recording,
                        'channels': ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']  # Channel names
                    })
            except Exception as e:
                print(f"Error in send_eeg_data: {e}")
        socketio.sleep(0.02)  # 50Hz update rate

if __name__ == '__main__':
    socketio.start_background_task(send_eeg_data)
    socketio.run(app, debug=True)