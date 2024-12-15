from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QGroupBox, QCheckBox, QScrollArea, QSlider, QLineEdit)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
from eeg_manager import EEGManager
import numpy as np
import mne
import io
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap
from brainflow.board_shim import BoardShim, BoardIds
from scipy.signal import butter, filtfilt

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
        # Create scroll area for control panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        scroll.setWidget(panel)
        
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
        
        # Task display
        self.task_label = QLabel("Current Task: None")
        self.instruction_label = QLabel("Instruction: ")
        self.progress_label = QLabel("Progress: 0/0")
        layout.addWidget(self.task_label)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.progress_label)
        
        # Status messages
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        layout.addWidget(QLabel("Status Messages:"))
        layout.addWidget(self.status_text)
        
        # Analysis controls
        self.analyze_btn = QPushButton("Analyze Session")
        self.analyze_btn.clicked.connect(self.analyze_session)
        layout.addWidget(self.analyze_btn)
        
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
        
        return scroll
        
    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create plot widget with 6 subplots (one for each channel)
        self.eeg_plot = pg.GraphicsLayoutWidget()
        self.eeg_curves = []
        channel_names = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
        
        # Create 6 subplots arranged vertically
        for i, name in enumerate(channel_names):
            # Add subplot
            plot = self.eeg_plot.addPlot(row=i, col=0)
            plot.setTitle(name)
            plot.setLabel('left', "μV")
            if i == len(channel_names) - 1:  # Only show x-axis label on bottom plot
                plot.setLabel('bottom', "Samples")
            else:
                plot.getAxis('bottom').hide()
            
            # Set y-axis range the same for all plots
            plot.enableAutoRange(axis='y')  # Enable auto-range by default
            plot.setAutoVisible(y=True)
            plot.setDownsampling(mode='peak')  # Add peak detection for better visualization
            plot.setClipToView(True)
            
            # Add curve to subplot
            curve = plot.plot(pen=self.channel_colors[name])
            self.eeg_curves.append(curve)
            
            # Link x-axes of all plots
            if i > 0:
                plot.setXLink(self.eeg_plot.getItem(0, 0))
        
        layout.addWidget(self.eeg_plot)
        
        # CSP Analysis Plot
        self.csp_plot = pg.GraphicsLayoutWidget()
        layout.addWidget(self.csp_plot)
        
        return panel
        
    def add_status_message(self, message):
        self.status_text.append(f"{message}")
        
    def start_session(self):
        try:
            session_num = self.session_num.value()
            self.eeg_manager.initialize_session(session_num)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_status_message(f"Started session {session_num}")
            self.update_session_buttons()  # Update session list
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
            self.progress_label.setText(
                f"Progress: {self.eeg_manager.current_task_index}/{len(self.eeg_manager.task_sequence)}"
            )
            # Start countdown
            self.countdown_remaining = 3
            self.countdown_timer.start(1000)  # 1 second intervals
        else:
            self.add_status_message("All tasks completed")
            self.stop_session()
            
    def update_plots(self):
        """Update real-time EEG plot"""
        if self.eeg_manager.board and self.eeg_manager.board.is_prepared():
            data = self.eeg_manager.get_current_data()
            if data is not None:
                # Apply filters
                data = self.apply_filters(data)
                
                # Get enabled channels and their indices
                enabled_channels = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
                enabled_indices = [i for i, ch in enumerate(enabled_channels) 
                                   if self.eeg_manager.channel_states[ch]]
                
                # Update only enabled channels
                data_idx = 0
                for i, curve in enumerate(self.eeg_curves):
                    if self.channel_checkboxes[enabled_channels[i]].isChecked():
                        curve.setData(data[data_idx])
                        data_idx += 1
                    
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
        # Create figure with subplots for each class
        plt.figure(figsize=(20, 15))
        
        for i, (pattern, class_name) in enumerate(zip(patterns, class_names)):
            plt.subplot(2, 3, i + 1)
            mne.viz.plot_topomap(pattern[:, 0], self.eeg_manager.raw_data.info, 
                                show=False, contours=0)
            plt.title(f'CSP Pattern for {class_name}')
        
        plt.tight_layout()
        
        # Convert to QPixmap and display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        buf.close()
        
        # Show in a new window
        label = QLabel()
        label.setPixmap(pixmap)
        
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.show()
        
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
        self.add_status_message(f"Recording: {self.time_remaining}s remaining")
        
        if self.time_remaining <= 0:
            self.recording_timer.stop()
            self.eeg_manager.stop_recording()
            self.start_break()
        
    def start_break(self):
        """Start break between tasks"""
        self.time_remaining = 3  # 3 second break
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