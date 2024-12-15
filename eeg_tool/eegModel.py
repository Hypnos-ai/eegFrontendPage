import time
from AdaptiveDQN_RLEEGNET import AdaptiveDQNRLEEGNET
import numpy as np
import time
import socket
from stable_baselines3 import DQN
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from mne.decoding import CSP
import joblib
import tensorflow as tf
import mne
import torch

MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.8
COMMAND_DURATION = 0.5
dqnutils = AdaptiveDQNRLEEGNET()
# Load the pre-trained DQN model
model = DQN.load("dqn_plasticity_final")
print("DQN model loaded.")

# Load the pre-trained CSP filters
csp_filters = joblib.load('csp_filters_ovr.pkl')
print("CSP filters loaded for real-time use.")

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')
print("Scaler loaded for real-time use.")

class MultiChannelCircularBuffer:
    def __init__(self, num_channels, buffer_size, window_size=1250, overlap_ratio=0.8):
        self.buffer = np.zeros((num_channels, buffer_size))
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.overlap_samples = int(window_size * overlap_ratio)
        self.step_size = window_size - self.overlap_samples
        self.index = 0
        self.last_window_end = 0
        
        print(f"Initialized circular buffer:")
        print(f"- Buffer size: {buffer_size} samples ({buffer_size/250:.1f}s)")
        print(f"- Window size: {window_size} samples ({window_size/250:.1f}s)")
        print(f"- Overlap: {overlap_ratio*100:.0f}% ({self.overlap_samples} samples)")
        print(f"- Step size: {self.step_size} samples ({self.step_size/250:.3f}s)")

    def add_data(self, data):
        """Add new data and return True if enough new samples for next window"""
        num_samples = data.shape[1]
        
        # Add data to buffer
        if self.index + num_samples <= self.buffer_size:
            self.buffer[:, self.index:self.index + num_samples] = data
        else:
            # Handle wrap-around
            end_index = (self.index + num_samples) % self.buffer_size
            self.buffer[:, self.index:] = data[:, :self.buffer_size - self.index]
            self.buffer[:, :end_index] = data[:, self.buffer_size - self.index:]
            
        self.index = (self.index + num_samples) % self.buffer_size
        
        # Check if we have enough new samples since last window
        samples_since_last = (self.index - self.last_window_end) % self.buffer_size
        return samples_since_last >= self.step_size

    def get_window(self):
        """Get the latest overlapping window of data"""
        # Update last window position
        self.last_window_end = self.index
        
        # Calculate start position for window
        start_idx = (self.index - self.window_size) % self.buffer_size
        
        if start_idx < self.index:
            return self.buffer[:, start_idx:self.index]
        else:
            return np.concatenate((
                self.buffer[:, start_idx:],
                self.buffer[:, :self.index]
            ), axis=1)


class SimulationConnection:
    def __init__(self, host='http://localhost:5000'):
        self.host = host
    
    def send_command(self, command):
        try:
            import requests
            response = requests.get(f"{self.host}/command/{command}")
            return response.json()['status'] == 'ok'
        except Exception as e:
            return False
    
    def close(self):
        pass

simulation = SimulationConnection()

class ArduinoConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.current_command = 'S'
        self.STOP_DURATION = 0.5
        
    def send_command(self, new_command):
        if new_command != self.current_command:
            try:
                # Send stop command
                print("Stopping for 1 second...")
                self._send_http_command('S')
                time.sleep(self.STOP_DURATION)
                
                # Send new command
                self._send_http_command(new_command)
                self.current_command = new_command
                print(f"Changed to new command: {new_command}")
                
            except Exception as e:
                print(f"Connection error: {e}")
    
    def _send_http_command(self, command):
        import requests
        try:
            url = f"http://{self.host}:{self.port}/{command}"
            requests.get(url, timeout=1)
        except Exception as e:
            print(f"HTTP request error: {e}")
    
    def close(self):
        self._send_http_command('S')
arduino = ArduinoConnection('192.168.1.168', 80)

def map_action_to_command(action):
    action_mapping = {
        0: 'L',
        1: 'R',
        2: 'F',
        3: 'B'
    }
    return action_mapping.get(action, 'S')

# Initialize components
num_channels = 6
buffer_size = 1250 * 10  # 10 seconds buffer
window_size = 1250       # 5 seconds window
samples_per_read = 1250  # Read 5 seconds of data at a time

# Initialize the buffer
eeg_buffer = MultiChannelCircularBuffer(
    num_channels=num_channels,
    buffer_size=buffer_size,
    window_size=window_size,
    overlap_ratio=0.65
)
predictions = []
# Set up BrainFlow
params = BrainFlowInputParams()
params.serial_port = 'COM9'  # Replace with your actual COM port
board = BoardShim(BoardIds.CYTON_BOARD.value, params)
board.prepare_session()
print("BrainFlow session prepared.")
board.start_stream()
print("Started data stream from board.")

def get_conservative_prediction(predictions, confidences, similarity_threshold=0.5):
    """
    When confidences are close (within threshold), pick the lower confidence prediction
    """
    if not confidences:  # Empty list check
        return None, None
    
    # Get the highest confidence and its index
    max_conf = max(confidences)
    max_idx = confidences.index(max_conf)
    
    # Look for confidences that are close to the maximum
    close_indices = [i for i, conf in enumerate(confidences) 
                    if (max_conf - conf) <= similarity_threshold]
    
    if len(close_indices) > 1:
        # If we have close confidences, pick the lower one
        min_conf_idx = min(close_indices, key=lambda i: confidences[i])
        return predictions[min_conf_idx], confidences[min_conf_idx]
    
    # If no close confidences, return the highest one
    return predictions[max_idx], confidences[max_idx]

def cleanup_resources():
    """
    Clean up board and connection resources
    """
    try:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
        arduino.close()
        print("Resources cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def main():
    try:
        # Initialize state variables
        prediction_history = []
        last_state = None
        last_command = 'S'
        last_command_time = time.time()
        start_time = time.time()
        model.exploration_rate = 2
        model.exploration_final_eps = 2
        while True:
            # Get new data (5 second chunks)
            data = board.get_current_board_data(samples_per_read)
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            eeg_data = data[eeg_channels[:6], :]
            

            # Add data and check if we have enough for a window
            if eeg_buffer.add_data(eeg_data):
                # Get window and process
                window_data = eeg_buffer.get_window().reshape(1, num_channels, -1)
                print(f"Processing window of shape: {window_data.shape}")
                
                # Filter data
                # data_downsampled = mne.filter.resample(window_data, up=2)  # Actually convert to 125Hz
                # filtered_data = mne.filter.filter_data(data_downsampled, sfreq=250, l_freq=0.5, h_freq=45, verbose=False)
                # Initialize CSP features list
                csp_features_list = []
                
                # Apply CSP transform for each class
                for class_id, csp in csp_filters.items():
                    transformed = csp.transform(window_data)
                    # if hasattr(raw, 'info'):  # Only plot if raw.info is available
                    #     csp.plot_patterns(info=raw.info, components=list(range(4)), ch_type='eeg')
                    csp_features_list.append(transformed)
                
                # Concatenate CSP features
                csp_ftrs = np.concatenate(csp_features_list, axis=1)
                print(f"CSP Features Shape: {csp_ftrs.shape}")
                
                # Extract features
                ftrs = dqnutils.featuresarray_load(data_array=csp_ftrs)
                print(f"Extracted Features Shape: {ftrs.shape}")
                
                predictions = []
                confidences = []

                for i in range(0, 20, 4):
                    ftrs_scaled = scaler.transform(ftrs.reshape(-1, ftrs.shape[-1])).reshape(ftrs.shape)
                    ftrs_scaled = ftrs_scaled[:, i:i+4, :]
                    
                    
                    # Get prediction
                    action, _ = model.predict(ftrs_scaled, deterministic=False)  # Changed this line
                    action = action[0]
                    # command = map_action_to_command(action)
                    # print(f"Mapped Command: {command} ")
                    # simulation.send_command(command)
                    # Calculate confidence
                    q_values = model.q_net(model.q_net.obs_to_tensor(ftrs_scaled)[0])[0].detach().numpy()
                    q_values = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values) + 1e-6)
                    
                    max_q = np.max(q_values)
                    other_q_values = q_values[q_values != max_q]
                    confidence = float(max_q - np.mean(other_q_values)) if len(other_q_values) > 0 else 0
                    
                    # Only add high confidence predictions
                    if confidence > 0.6:  # You can adjust this threshold
                        predictions.append(int(action))
                        confidences.append(confidence)

                # Get conservative prediction
                action, confidence = get_conservative_prediction(predictions, confidences)
                if action is not None:
                    command = map_action_to_command(action)
                    print(f"Mapped Command: {command} {predictions}{confidences}")
                else:
                    command = 'S'  # Default to stop if no confident predictions
                simulation.send_command(command)        
                arduino.send_command(command)
                
            time.sleep(0.02)  # 20ms sleep

    except KeyboardInterrupt:
        print("\nStopped by user.")    
    finally:
        simulation.close()
        board.stop_stream()
        board.release_session()
        simulation.send_command('S')
        arduino.send_command('S')
        

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_resources()