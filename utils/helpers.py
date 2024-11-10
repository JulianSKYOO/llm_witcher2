import numpy as np
import time

class LengthSampler:
    """Sample length for text generation"""
    def __init__(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self):
        return np.random.randint(self.min_length, self.max_length)

def collator(data):
    """Collate data for batch processing"""
    return dict((key, [d[key] for d in data]) for key in data[0])

def format_time(seconds):
    """Format time in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

class PerformanceTracker:
    """Track performance metrics"""
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'training_time': [],
            'model_load_time': [],
            'toxicity_scores': []
        }
    
    def add_metric(self, metric_type, value):
        if metric_type in self.metrics:
            self.metrics[metric_type].append(value)
    
    def get_average(self, metric_type):
        if metric_type in self.metrics and self.metrics[metric_type]:
            return np.mean(self.metrics[metric_type])
        return 0.0
    
    def get_latest(self, metric_type):
        if metric_type in self.metrics and self.metrics[metric_type]:
            return self.metrics[metric_type][-1]
        return 0.0

class Timer:
    """Context manager for timing operations"""
    def __init__(self, tracker=None, operation=None):
        self.tracker = tracker
        self.operation = operation
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.duration = time.time() - self.start
        if self.tracker and self.operation:
            self.tracker.add_metric(self.operation, self.duration)