"""
Base metrics class for monitoring training and pruning
"""
import abc
from typing import Dict, Any, List, Optional
import logging

class BaseMetric(abc.ABC):
    """
    Abstract base class for metrics collection
    """
    
    def __init__(self):
        """Initialize the metric collector"""
        self.values = []
        
    @abc.abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """
        Collect the current metric value
        
        Args:
            **kwargs: Additional parameters for collection
            
        Returns:
            Dictionary with metric values
        """
        pass
    
    def reset(self):
        """Reset all collected values"""
        self.values = []
        
    def get_values(self) -> List:
        """Get all collected values"""
        return self.values
    
    def get_latest(self) -> Optional[Any]:
        """Get the most recent value"""
        if self.values:
            return self.values[-1]
        return None
    
    def get_average(self) -> Optional[float]:
        """Get average of collected values"""
        if not self.values:
            return None
        return sum(self.values) / len(self.values)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the collected metrics
        
        Returns:
            Dictionary with metric summary
        """
        return {
            'latest': self.get_latest(),
            'average': self.get_average(),
            'count': len(self.values)
        }
