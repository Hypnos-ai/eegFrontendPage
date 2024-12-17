from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QGroupBox, QCheckBox, QScrollArea, QSlider, QLineEdit, QProgressBar, QFrame, QTabWidget)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from eeg_manager import EEGManager
import numpy as np
import mne
import io
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from brainflow.board_shim import BoardShim, BoardIds
from scipy.signal import butter, filtfilt
import os
from stable_baselines3.common.callbacks import BaseCallback
from AdaptiveDQN_RLEEGNET import AdaptiveDQNRLEEGNET
import scipy.signal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize EEG Manager
        self.eeg_manager = EEGManager()
        
        # Disable analyze button until session is selected
        self.analyze_btn = None  # Will be set in create_control_panel
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        try:
            # Try to initialize board, but continue if it fails
            self.eeg_manager.initialize_board()
        except Exception as e:
            print(f"Board initialization failed: {e}")
            # Continue without board connection
        
        # Create control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel, stretch=1)
        
        # Create visualization panel
        viz_panel = self.create_visualization_panel()
        layout.addWidget(viz_panel, stretch=2)
        
        # Setup update timer for real-time plotting
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(20)  # 50Hz update rate for smoother display
        
        # Timer setup for task management
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.recording_timeout)
        self.break_timer = QTimer()
        self.break_timer.timeout.connect(self.break_timeout)
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.countdown_timeout)
        
        self.time_remaining = 0
        self.countdown_remaining = 0
        
        # Add cleanup on close
        self.setAttribute(Qt.WA_DeleteOnClose)
        
    def create_control_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Add task display panel at the top
        task_panel = self.create_task_display_panel()
        layout.addWidget(task_panel)
        
        # Session selection
        session_group = QGroupBox("Session Management")
        session_layout = QVBoxLayout(session_group)
        
        # New session controls
        new_session_widget = QWidget()
        new_session_layout = QHBoxLayout(new_session_widget)
        self.session_num = QSpinBox()
        self.session_num.setMinimum(1)
        new_session_layout.addWidget(QLabel("New Session:"))
        new_session_layout.addWidget(self.session_num)
        session_layout.addWidget(new_session_widget)
        
        # Existing sessions
        self.session_buttons = QWidget()
        self.session_buttons_layout = QVBoxLayout(self.session_buttons)
        self.update_session_buttons()
        session_layout.addWidget(self.session_buttons)
        
        layout.addWidget(session_group)
        
        # Session Configuration
        config_group = QGroupBox("Session Configuration")
        config_group.setStyleSheet("""
            QGroupBox {
                background-color: #f5f5f5;
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                color: #2196F3;
            }
        """)
        config_layout = QVBoxLayout(config_group)
        
        # Tasks per type control
        tasks_widget = QWidget()
        tasks_layout = QHBoxLayout(tasks_widget)
        tasks_layout.addWidget(QLabel("Tasks per type:"))
        self.tasks_per_type = QSpinBox()
        self.tasks_per_type.setRange(1, 50)
        self.tasks_per_type.setValue(21)  # Default value
        self.tasks_per_type.setToolTip("Number of trials for each task type")
        tasks_layout.addWidget(self.tasks_per_type)
        config_layout.addWidget(tasks_widget)
        
        # Break duration control
        break_widget = QWidget()
        break_layout = QHBoxLayout(break_widget)
        break_layout.addWidget(QLabel("Break duration (sec):"))
        self.break_duration = QSpinBox()
        self.break_duration.setRange(1, 10)
        self.break_duration.setValue(3)  # Default value
        self.break_duration.setToolTip("Duration of break between trials")
        break_layout.addWidget(self.break_duration)
        config_layout.addWidget(break_widget)
        
        # Trial duration display (constant)
        trial_widget = QWidget()
        trial_layout = QHBoxLayout(trial_widget)
        trial_layout.addWidget(QLabel("Trial duration: 5 seconds (fixed)"))
        config_layout.addWidget(trial_widget)
        
        layout.addWidget(config_group)
        
        # Channel controls
        channel_group = QGroupBox("EEG Channels")
        channel_layout = QVBoxLayout(channel_group)
        self.channel_checkboxes = {}
        self.channel_names = {}  # Store channel name inputs
        self.channel_colors = {
            'FCz': '#1f77b4',  # Blue
            'C3': '#ff7f0e',   # Orange
            'Cz': '#2ca02c',   # Green
            'CPz': '#d62728',  # Red
            'C2': '#9467bd',   # Purple
            'C4': '#8c564b'    # Brown
        }
        
        # Add header for channel configuration
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.addWidget(QLabel("Channel"), stretch=1)
        header_layout.addWidget(QLabel("Name"), stretch=1)
        header_layout.addWidget(QLabel("Enable"), stretch=1)
        channel_layout.addWidget(header)
        
        for channel, color in self.channel_colors.items():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            
            # Color indicator
            color_label = QLabel("■")
            color_label.setStyleSheet(f"color: {color}; font-size: 20px;")
            row_layout.addWidget(color_label)
            
            # Channel name input
            name_input = QLineEdit(channel)  # Default to original name
            name_input.setPlaceholderText("Enter channel name")
            name_input.textChanged.connect(lambda text, ch=channel: self.update_channel_name(ch, text))
            self.channel_names[channel] = name_input
            row_layout.addWidget(name_input)
            
            # Channel checkbox
            checkbox = QCheckBox(channel)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_channel_visibility)
            self.channel_checkboxes[channel] = checkbox
            row_layout.addWidget(checkbox)
            
            channel_layout.addWidget(row)
            
        # Add Apply button for channel configuration
        apply_btn = QPushButton("Apply Channel Configuration")
        apply_btn.clicked.connect(self.apply_channel_config)
        channel_layout.addWidget(apply_btn)
        
        layout.addWidget(channel_group)
        
        # Session controls
        self.start_btn = QPushButton("Start Session")
        self.start_btn.clicked.connect(self.start_session)
        self.stop_btn = QPushButton("Stop Session")
        self.stop_btn.clicked.connect(self.stop_session)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        
        # Analysis controls
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analyze_btn = QPushButton("Analyze Session (CSP)")
        self.analyze_btn.clicked.connect(self.analyze_session)
        analysis_layout.addWidget(self.analyze_btn)
        
        # DQN Training controls
        self.train_dqn_btn = QPushButton("Train DQN-LSTM Model")
        self.train_dqn_btn.clicked.connect(self.train_dqn_model)
        analysis_layout.addWidget(self.train_dqn_btn)
        
        # Training progress
        self.training_progress = QTextEdit()
        self.training_progress.setReadOnly(True)
        self.training_progress.setMaximumHeight(100)
        analysis_layout.addWidget(QLabel("Training Progress:"))
        analysis_layout.addWidget(self.training_progress)
        
        layout.addWidget(analysis_group)
        
        # Add signal controls
        signal_group = QGroupBox("Signal Controls")
        signal_layout = QVBoxLayout(signal_group)
        
        # Speed control
        speed_widget = QWidget()
        speed_layout = QHBoxLayout(speed_widget)
        speed_layout.addWidget(QLabel("Update Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.update_plot_speed)
        speed_layout.addWidget(self.speed_slider)
        signal_layout.addWidget(speed_widget)
        
        # Filter controls
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.addWidget(QLabel("Bandpass Filter:"))
        self.lowcut = QDoubleSpinBox()
        self.lowcut.setRange(0.1, 100.0)
        self.lowcut.setDecimals(1)
        self.lowcut.setSingleStep(0.1)
        self.lowcut.setValue(0.5)  # Default OpenBCI lowcut
        
        self.highcut = QDoubleSpinBox()
        self.highcut.setRange(1.0, 100.0)
        self.highcut.setDecimals(1)
        self.highcut.setSingleStep(0.1)
        self.highcut.setValue(45.0)  # Default OpenBCI highcut
        
        filter_layout.addWidget(QLabel("Low:"))
        filter_layout.addWidget(self.lowcut)
        filter_layout.addWidget(QLabel("High:"))
        filter_layout.addWidget(self.highcut)
        signal_layout.addWidget(filter_widget)
        
        layout.addWidget(signal_group)
        
        # Add Status Messages section
        status_group = QGroupBox("Status Messages")
        status_group.setStyleSheet("""
            QGroupBox {
                background-color: #f5f5f5;
                border: 2px solid #9e9e9e;
                border-radius: 5px;
                margin-top: 1ex;
                font-size: 14px;
            }
            QGroupBox::title {
                color: #424242;
                subcontrol-origin: margin;
                padding: 0 3px;
            }
        """)
        status_layout = QVBoxLayout(status_group)
        
        # Create status text widget
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 5px;
                font-family: monospace;
            }
        """)
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        # Set the panel as the scroll area widget
        scroll.setWidget(panel)
        return scroll
        
    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create plot widget with 6 subplots (one for each channel)
        self.eeg_plot = pg.GraphicsLayoutWidget()
        self.eeg_plot.setBackground('w')
        self.eeg_curves = []
        channel_names = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
        
        # Create 6 subplots arranged vertically
        for i, name in enumerate(channel_names):
            # Add subplot
            plot = self.eeg_plot.addPlot(row=i, col=0)
            plot.setTitle(name, color='k')
            plot.setLabel('left', "μV", color='k')
            if i == len(channel_names) - 1:
                plot.setLabel('bottom', "Samples", color='k')
            else:
                plot.getAxis('bottom').hide()
            
            plot.enableAutoRange(axis='y')
            plot.setAutoVisible(y=True)
            plot.setDownsampling(mode='peak')
            plot.setClipToView(True)
            plot.showGrid(x=True, y=True, alpha=0.3)
            
            # Add single curve for filtered data
            curve = plot.plot(pen=self.channel_colors[name])
            self.eeg_curves.append(curve)
            
            if i > 0:
                plot.setXLink(self.eeg_plot.getItem(0, 0))
        
        layout.addWidget(self.eeg_plot)
        return panel
        
    def add_status_message(self, message):
        self.status_text.append(f"{message}")
        
    def start_session(self):
        try:
            session_num = self.session_num.value()
            
            # Update task configuration
            self.eeg_manager.task_names = {
                "Right_Hand": ["Imagine or perform right hand movement", self.tasks_per_type.value()],
                "Left_Hand": ["Imagine or perform left hand movement", self.tasks_per_type.value()],
                "Blinking": ["Blink your eyes at a comfortable pace", self.tasks_per_type.value()],
                "Jaw_Clenching": ["Clench your jaw with moderate force", self.tasks_per_type.value()],
                "Relax": ["Relax your body and mind, do nothing", self.tasks_per_type.value()]
            }
            
            self.eeg_manager.initialize_session(session_num)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_status_message(f"Started session {session_num}")
            self.update_session_buttons()
            self.get_next_task()
        except Exception as e:
            self.add_status_message(f"Error starting session: {str(e)}")
            
    def stop_session(self):
        try:
            self.eeg_manager.stop_session()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.add_status_message("Session stopped")
        except Exception as e:
            self.add_status_message(f"Error stopping session: {str(e)}")
            
    def get_next_task(self):
        task = self.eeg_manager.get_next_task()
        if task:
            task_name, instruction = task
            self.task_label.setText(f"Current Task: {task_name}")
            self.instruction_label.setText(f"Instruction: {instruction}")
            
            # Update progress
            current = self.eeg_manager.current_task_index
            total = len(self.eeg_manager.task_sequence)
            self.progress_label.setText(f"Progress: {current}/{total}")
            self.progress_bar.setValue(int((current / total) * 100))
            
            # Update sequence display
            sequence_text = "Upcoming Tasks:\n"
            remaining_tasks = self.eeg_manager.task_sequence[current:]
            for i, (task, instr) in enumerate(remaining_tasks, start=current+1):
                sequence_text += f"{i}. {task}\n"
            self.sequence_list.setText(sequence_text)
            
            # Start countdown
            self.countdown_remaining = 3
            self.countdown_timer.start(1000)
        else:
            self.add_status_message("All tasks completed")
            self.stop_session()
            
    def update_plots(self):
        """Update real-time EEG plot"""
        try:
            if hasattr(self.eeg_manager, 'board') and self.eeg_manager.board and self.eeg_manager.board.is_prepared():
                data = self.eeg_manager.get_current_data()
                if data is not None and len(self.eeg_curves) > 0:
                    # Apply filters
                    filtered_data = self.apply_filters(data)
                    
                    # Get enabled channels and their indices
                    enabled_channels = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
                    enabled_indices = [i for i, ch in enumerate(enabled_channels) 
                                   if self.eeg_manager.channel_states.get(ch, False)]
                    
                    # Update curves
                    data_idx = 0
                    for i, curve in enumerate(self.eeg_curves):
                        if i < len(enabled_channels) and enabled_channels[i] in self.channel_checkboxes:
                            if self.channel_checkboxes[enabled_channels[i]].isChecked():
                                if data_idx < len(filtered_data):
                                    curve.setData(filtered_data[data_idx])
                                    data_idx += 1
                            
        except Exception as e:
            print(f"Error updating plots: {e}")
            
    def analyze_session(self):
        try:
            self.add_status_message("Starting analysis...")
            if self.eeg_manager.current_session_num is None:
                raise ValueError("No session selected. Please select a session first.")
            patterns, class_names = self.eeg_manager.analyze_session(self.eeg_manager.current_session_num)
            self.display_csp_patterns(patterns, class_names)
            self.add_status_message("Analysis complete")
        except Exception as e:
            self.add_status_message(f"Analysis error: {str(e)}")
            
    def get_channel_positions(self):
        """Get channel positions in 2D for topographic plotting"""
        # Standard 10-20 positions for our channels
        positions = {
            'FCz': [0, 0.25],
            'C3': [-0.5, 0],
            'Cz': [0, 0],
            'CPz': [0, -0.25],
            'C2': [0.25, 0],
            'C4': [0.5, 0]
        }
        return positions

    def display_csp_patterns(self, patterns, class_names):
        # Create a new window
        analysis_window = QWidget()
        analysis_window.setWindowTitle("CSP Analysis Results")
        analysis_window.setGeometry(100, 100, 1200, 800)
        layout = QVBoxLayout(analysis_window)
        
        # Create matplotlib figure
        plt.style.use('default')  # Reset style
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        axes = axes.ravel()  # Flatten axes array for easier indexing
        
        for i, (pattern, class_name) in enumerate(zip(patterns, class_names)):
            ax = axes[i]
            mne.viz.plot_topomap(pattern[:, 0], self.eeg_manager.raw_data.info, 
                                axes=ax, show=False, contours=0,
                                cmap='RdBu_r', sensors=True,
                                outlines='head')
            ax.set_title(f'CSP Pattern for {class_name}')
        
        plt.tight_layout()
        
        # Convert matplotlib figure to QPixmap
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        buf.close()
        
        # Show in a new window
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(1000, 800)  # Set minimum size
        scroll.setWidget(label)
        layout.addWidget(scroll)
        
        # Show window
        analysis_window.show()
        analysis_window.raise_()  # Bring window to front
        analysis_window.activateWindow()  # Activate the window
        
        # Store reference to prevent garbage collection
        self._analysis_window = analysis_window
        
    def countdown_timeout(self):
        """Handle countdown before recording"""
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            self.add_status_message(f"Starting in {self.countdown_remaining}...")
        else:
            self.countdown_timer.stop()
            self.start_recording()
        
    def start_recording(self):
        """Start recording current task"""
        if self.eeg_manager.start_recording():
            self.time_remaining = 5  # 5 second trials
            self.recording_timer.start(1000)  # 1 second intervals
            self.add_status_message("Recording started")
        
    def recording_timeout(self):
        """Handle recording timer"""
        self.time_remaining -= 1
        self.timer_label.setText(f"Time Remaining: {self.time_remaining}s")
        self.timer_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FF5722;
            padding: 10px;
            background-color: #FBE9E7;
            border-radius: 5px;
        """)
        
        # Update progress bar
        total_tasks = len(self.eeg_manager.task_sequence)
        current_task = self.eeg_manager.current_task_index
        progress = int((current_task / total_tasks) * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Progress: {current_task}/{total_tasks}")
        
        if self.time_remaining <= 0:
            self.recording_timer.stop()
            self.eeg_manager.stop_recording()
            self.start_break()
        
    def start_break(self):
        """Start break between tasks"""
        self.time_remaining = self.break_duration.value()  # Use configured break duration
        self.break_timer.start(1000)
        self.add_status_message("Break started")
        
    def break_timeout(self):
        """Handle break timer"""
        self.time_remaining -= 1
        self.add_status_message(f"Break: {self.time_remaining}s remaining")
        
        if self.time_remaining <= 0:
            self.break_timer.stop()
            self.get_next_task()
        
    def toggle_board_connection(self):
        if self.eeg_manager.is_streaming:
            # Stop streaming
            self.eeg_manager.stop_stream()
            self.connect_btn.setText("Connect Board")
            self.start_btn.setEnabled(False)
            self.add_status_message("Board disconnected")
        else:
            # Start streaming
            if self.eeg_manager.start_stream():
                self.connect_btn.setText("Disconnect Board")
                self.start_btn.setEnabled(True)
                self.add_status_message("Board connected")
            else:
                self.add_status_message("Failed to connect board")
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.eeg_manager.is_streaming:
            self.eeg_manager.stop_stream()
        event.accept()
        
    def update_channel_visibility(self):
        """Update which channels are visible in the plot"""
        for i, (channel, checkbox) in enumerate(self.channel_checkboxes.items()):
            # Update EEG Manager channel state
            self.eeg_manager.set_channel_state(channel, checkbox.isChecked())
            # Update plot visibility
            plot = self.eeg_plot.getItem(i, 0)
            plot.setVisible(checkbox.isChecked())
        
    def update_session_buttons(self):
        """Update the list of session buttons"""
        # Clear existing buttons
        while self.session_buttons_layout.count():
            item = self.session_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Get existing sessions from annotations file
        try:
            with open(self.eeg_manager.annotation_file, 'r') as f:
                content = f.read()
                sessions = sorted(list(set([
                    int(x.replace('session', '')) 
                    for x in content.split('\n') 
                    if x.strip().startswith('session')
                ])))
                
            if sessions:
                self.session_buttons_layout.addWidget(QLabel("Existing Sessions:"))
                for session in sessions:
                    btn = QPushButton(f"Session {session}")
                    btn.clicked.connect(lambda checked, s=session: self.select_session(s))
                    self.session_buttons_layout.addWidget(btn)
        except Exception as e:
            print(f"Error loading sessions: {e}")
            
    def select_session(self, session_num):
        """Select an existing session for analysis"""
        self.eeg_manager.current_session_num = session_num
        self.add_status_message(f"Selected Session {session_num}")
        # Enable analyze button only when a session is selected
        self.analyze_btn.setEnabled(True)
        
    def update_plot_speed(self):
        """Update the plot refresh rate"""
        speed = self.speed_slider.value()
        interval = int(1000 / speed)  # Convert to milliseconds
        self.update_timer.setInterval(interval)
        
    def apply_filters(self, data):
        """Apply bandpass filter to data"""
        if data is None or data.size < 30:  # Add minimum size check
            return data
            
        lowcut = self.lowcut.value()
        highcut = self.highcut.value()
        
        if lowcut >= highcut:
            return data
            
        # Use scipy's butterworth filter
        nyquist = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value) / 2.0
        
        # Create butterworth bandpass filter
        b, a = butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
        
        # Add padding for short signals
        padlen = 3 * max(len(a), len(b))
        if data.shape[1] <= padlen:
            return data
        
        # Apply filter
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i, :] = filtfilt(b, a, data[i, :])
        
        return filtered_data
        
    def update_channel_name(self, original_name, new_name):
        """Store updated channel name"""
        if new_name.strip():  # Only update if not empty
            self.channel_names[original_name].setStyleSheet("")
        else:
            self.channel_names[original_name].setStyleSheet("background-color: #ffebee")
            
    def apply_channel_config(self):
        """Apply channel name changes"""
        # Validate all names are unique and not empty
        new_names = [input.text().strip() for input in self.channel_names.values()]
        if len(set(new_names)) != len(new_names) or '' in new_names:
            self.add_status_message("Error: Channel names must be unique and not empty")
            return
            
        # Update plot titles and channel mapping
        for i, (original_name, input_widget) in enumerate(self.channel_names.items()):
            new_name = input_widget.text().strip()
            plot = self.eeg_plot.getItem(i, 0)
            plot.setTitle(new_name)
            
        self.add_status_message("Channel configuration updated")
        
    def train_dqn_model(self):
        """Train DQN-LSTM model on the current session"""
        try:
            if self.eeg_manager.current_session_num is None:
                raise ValueError("No session selected. Please select a session first.")
                
            self.training_progress.clear()
            self.training_progress.append("Initializing DQN training...")
            
            # First combine all FIF files
            self.training_progress.append("Combining FIF files...")
            try:
                self.eeg_manager.combine_fif_files()
                self.training_progress.append("Files combined successfully")
            except Exception as e:
                raise Exception(f"Error combining files: {str(e)}")
            
            # Initialize the model
            dqn_model = AdaptiveDQNRLEEGNET()
            
            # Setup EEG data
            combined_file = os.path.join(self.eeg_manager.formatted_dir, 'S03.fif')  # Use same filename as in combine_fif_files
            raw, events = dqn_model.setup_eeg(combined_file)
            self.training_progress.append("Data loaded successfully")
            
            # Process epochs
            X_train, X_test, y_train, y_test = dqn_model.process_epochs()
            self.training_progress.append(f"Processed {len(X_train)} training samples")
            
            # Create environment
            env_class = dqn_model.create_environment()
            env = env_class()
            
            # Setup and train model
            model, callbacks = dqn_model.setup_training(env)
            
            # Add custom callback for GUI updates
            class GUICallback(BaseCallback):
                def __init__(self_, gui, verbose=0):
                    super().__init__(verbose)
                    self_.gui = gui
                
                def _on_step(self_):
                    if self_.n_calls % 1000 == 0:
                        self_.gui.training_progress.append(f"Training step: {self_.n_calls}")
                    return True
            
            callbacks.append(GUICallback(self))
            
            # Train model
            self.training_progress.append("Starting training...")
            model.learn(total_timesteps=2500, callback=callbacks)
            
            # Save model
            model_path = os.path.join(self.eeg_manager.processed_dir, f'dqn_model_session{self.eeg_manager.current_session_num}.zip')
            model.save(model_path)
            
            self.training_progress.append("Training complete!")
            self.training_progress.append(f"Model saved to: {model_path}")
            
        except Exception as e:
            self.training_progress.append(f"Training error: {str(e)}")
        
    def create_task_display_panel(self):
        task_panel = QGroupBox("Current Session")
        task_panel.setStyleSheet("""
            QGroupBox {
                background-color: #f0f0f0;
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 1ex;
                font-size: 14px;
            }
            QGroupBox::title {
                color: #2196F3;
                subcontrol-origin: margin;
                padding: 0 3px;
            }
        """)
        
        layout = QVBoxLayout(task_panel)
        
        # Session Status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        status_frame.setStyleSheet("background-color: white; padding: 10px;")
        status_layout = QVBoxLayout(status_frame)
        
        # Current Task Display
        self.task_label = QLabel("Current Task: None")
        self.task_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #1976D2;
            padding: 10px;
            background-color: #E3F2FD;
            border-radius: 5px;
        """)
        
        # Instruction Display
        self.instruction_label = QLabel("Instruction: ")
        self.instruction_label.setStyleSheet("""
            font-size: 14px;
            color: #424242;
            padding: 10px;
            background-color: #F5F5F5;
            border-radius: 5px;
        """)
        self.instruction_label.setWordWrap(True)
        
        # Progress Display
        progress_container = QWidget()
        progress_layout = QHBoxLayout(progress_container)
        
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_label.setStyleSheet("color: #616161; font-size: 14px;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #E0E0E0;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setMaximum(100)
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        # Timer Display
        self.timer_label = QLabel("Time Remaining: --")
        self.timer_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FF5722;
            padding: 10px;
            background-color: #FBE9E7;
            border-radius: 5px;
        """)
        
        # Add all elements to status frame
        status_layout.addWidget(self.task_label)
        status_layout.addWidget(self.instruction_label)
        status_layout.addWidget(progress_container)
        status_layout.addWidget(self.timer_label)
        
        # Task Sequence Display
        sequence_frame = QFrame()
        sequence_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        sequence_frame.setStyleSheet("background-color: white; padding: 10px;")
        sequence_layout = QVBoxLayout(sequence_frame)
        
        sequence_label = QLabel("Task Sequence")
        sequence_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1976D2;")
        self.sequence_list = QTextEdit()
        self.sequence_list.setReadOnly(True)
        self.sequence_list.setStyleSheet("""
            background-color: #FAFAFA;
            border: 1px solid #E0E0E0;
            border-radius: 5px;
            padding: 5px;
        """)
        
        sequence_layout.addWidget(sequence_label)
        sequence_layout.addWidget(self.sequence_list)
        
        # Add frames to main layout
        layout.addWidget(status_frame)
        layout.addWidget(sequence_frame)
        
        return task_panel

    def update_filter_response(self):
        """Update the filter frequency response plot"""
        try:
            fs = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
            nyquist = fs / 2.0
            
            lowcut = self.lowcut.value()
            highcut = self.highcut.value()
            
            if lowcut >= highcut:
                return
            
            # Generate frequency response
            order = 4
            b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
            
            # Calculate frequency response
            w, h = scipy.signal.freqz(b, a)
            
            # Convert to Hz and dB
            frequencies = w * nyquist / np.pi
            magnitude_db = 20 * np.log10(np.abs(h))
            
            # Update plot
            self.filter_plot.clear()
            self.filter_plot.plot(frequencies, magnitude_db, pen='b')
            
            # Add cutoff frequency markers
            self.filter_plot.addLine(x=lowcut, pen='r')
            self.filter_plot.addLine(x=highcut, pen='r')
            
        except Exception as e:
            print(f"Error updating filter response: {str(e)}")