class MultiChannelCircularBuffer:
    def __init__(self, num_channels, buffer_size, window_size, overlap_ratio=0.5):
        self.num_channels = num_channels
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap_samples = int(window_size * overlap_ratio)
        self.buffer = np.zeros((num_channels, buffer_size))
        self.index = 0
        
    def add_data(self, new_data):
        """
        Add new data to the buffer and return True if enough data for a window
        """
        samples_to_add = new_data.shape[1]
        if samples_to_add > self.buffer_size:
            new_data = new_data[:, -self.buffer_size:]
            samples_to_add = self.buffer_size
            
        # Roll buffer to make space for new data
        self.buffer = np.roll(self.buffer, -samples_to_add, axis=1)
        self.buffer[:, -samples_to_add:] = new_data
        
        self.index += samples_to_add
        return self.index >= self.window_size
    
    def get_window(self):
        """
        Get the latest window of data
        """
        return self.buffer[:, -self.window_size:]
    
    def reset(self):
        """
        Reset the buffer
        """
        self.buffer.fill(0)
        self.index = 0 