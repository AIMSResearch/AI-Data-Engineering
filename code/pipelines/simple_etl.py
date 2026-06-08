"""
Simple ETL pipeline skeleton for AI data engineering.
"""

import pandas as pd
from datetime import datetime
from typing import Callable, Dict, Any


class SimplePipeline:
    """Basic ETL pipeline framework."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.status = 'pending'
        self.data = None
        self.metrics = {}
    
    def extract(self, source: str, **kwargs) -> pd.DataFrame:
        """Extract data from source."""
        self.start_time = datetime.now()
        print(f"[{self.name}] Extracting data from {source}...")
        
        if source.endswith('.csv'):
            self.data = pd.read_csv(source, **kwargs)
        elif source.endswith('.json'):
            self.data = pd.read_json(source, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {source}")
        
        self.metrics['rows_extracted'] = len(self.data)
        print(f"✓ Extracted {len(self.data)} rows")
        return self.data
    
    def transform(self, func: Callable) -> pd.DataFrame:
        """Apply transformation function."""
        if self.data is None:
            raise ValueError("No data to transform. Run extract() first.")
        
        print(f"[{self.name}] Applying transformation...")
        self.data = func(self.data)
        self.metrics['rows_after_transform'] = len(self.data)
        print(f"✓ Transformation complete, {len(self.data)} rows remain")
        return self.data
    
    def load(self, destination: str, **kwargs) -> None:
        """Load data to destination."""
        if self.data is None:
            raise ValueError("No data to load. Run extract() and transform() first.")
        
        print(f"[{self.name}] Loading data to {destination}...")
        
        if destination.endswith('.csv'):
            self.data.to_csv(destination, index=False, **kwargs)
        elif destination.endswith('.json'):
            self.data.to_json(destination, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {destination}")
        
        self.end_time = datetime.now()
        self.status = 'success'
        duration = (self.end_time - self.start_time).total_seconds()
        self.metrics['duration_seconds'] = duration
        
        print(f"✓ Data loaded to {destination}")
        print(f"✓ Pipeline completed in {duration:.2f}s")
    
    def run(self, source: str, transform_func: Callable, destination: str, **kwargs) -> Dict[str, Any]:
        """Execute full ETL pipeline."""
        try:
            self.extract(source, **kwargs)
            self.transform(transform_func)
            self.load(destination, **kwargs)
            return {'status': 'success', 'metrics': self.metrics}
        except Exception as e:
            self.status = 'failed'
            print(f"❌ Pipeline failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
