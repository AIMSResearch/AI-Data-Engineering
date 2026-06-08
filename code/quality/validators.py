"""
Data validation utilities for AI data engineering pipelines.
"""

import pandas as pd
from typing import Dict, List, Tuple


class DataValidator:
    """Validates data quality at various pipeline stages."""
    
    def __init__(self):
        self.validation_results = []
    
    def check_nulls(self, df: pd.DataFrame, allowed_columns: List[str] = None) -> Tuple[bool, Dict]:
        """Check for null values in dataframe."""
        null_counts = df.isnull().sum()

        if allowed_columns:
            # Some pipelines tolerate nulls in optional fields but not in the
            # columns that anchor joins, labels, or compliance rules.
            null_counts = null_counts[null_counts.index.isin(allowed_columns)]
        
        has_nulls = null_counts.any()
        result = {
            'check': 'null_values',
            'passed': not has_nulls,
            'details': null_counts[null_counts > 0].to_dict() if has_nulls else {}
        }
        self.validation_results.append(result)
        return not has_nulls, result
    
    def check_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> Tuple[bool, Dict]:
        """Check for duplicate rows."""
        duplicates = df.duplicated(subset=subset).sum()
        
        result = {
            'check': 'duplicates',
            'passed': duplicates == 0,
            'duplicate_count': duplicates
        }
        self.validation_results.append(result)
        return duplicates == 0, result
    
    def check_schema(self, df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> Tuple[bool, Dict]:
        """Check if dataframe matches expected schema."""
        issues = {}

        for col, expected_type in expected_dtypes.items():
            if col not in df.columns:
                issues[col] = f"Column missing"
            elif str(df[col].dtype) != expected_type:
                issues[col] = f"Expected {expected_type}, got {df[col].dtype}"
        
        passed = len(issues) == 0
        result = {
            'check': 'schema',
            'passed': passed,
            'issues': issues
        }
        self.validation_results.append(result)
        return passed, result
    
    def check_value_range(self, df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None) -> Tuple[bool, Dict]:
        """Check if column values fall within expected range."""
        out_of_range = 0

        # Keep the range logic explicit so notebooks can extend it into source-
        # specific business rules rather than hiding the check behind a library.
        if min_val is not None:
            out_of_range += (df[column] < min_val).sum()
        if max_val is not None:
            out_of_range += (df[column] > max_val).sum()
        
        result = {
            'check': f'value_range_{column}',
            'passed': out_of_range == 0,
            'out_of_range_count': out_of_range,
            'range': (min_val, max_val)
        }
        self.validation_results.append(result)
        return out_of_range == 0, result
    
    def get_report(self) -> Dict:
        """Generate validation report."""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r['passed'])

        # A compact summary is easier to compare across notebook scenarios than
        # a long stream of per-check print statements.
        return {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': total_checks - passed_checks,
            'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            'details': self.validation_results
        }
