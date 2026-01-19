
import os
import logging
import argparse
import json
from typing import Dict, Any

DEFAULT_CONFIG = "config.json"
logger = logging.getLogger("eagleiLogger")

class ConfigManager:
    """Manages configuration loading and merging from file and CLI arguments."""
    
    def __init__(self, config_file: str = DEFAULT_CONFIG):
        """
        Initialize ConfigManager and load configuration.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config = self._load_default_config()
        if os.path.exists(config_file):
            self._load_config_file(config_file)
            logger.info(f"Loaded configuration from {config_file}")
        else:
            logger.warning(f"Config file {config_file} not found. Using defaults.")
    
    @staticmethod
    def _load_default_config() -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "processing_parameters": {
                "state": "Florida",
                "county": "Miami-Dade",
                "start_year": 2014,
                "end_year": 2024
            },

            "data_cleaning_parameters": {
                "min_customers_before_gap": 20,
                "min_customers_after_gap": 2,
                "max_gap_minutes": 1440,
                "use_auto_gap_rank_threshold": True,
                "gap_rank_threshold_quantile": 0.4,

                "events_customer_threshold":30
            },

            "processing_options": {
                "fetch_weather_data": True,
                "clean_weather_data": True,
                "clean_outage_data": True,
                "align_data": True,
                "skip_if_exists": True
            },

            "weather_variables": {
                "temperature": "tmpf",
                "wind_speed": "sknt",
                "wind_gust": "gust",
                "precipitation": "p01i"
            },

            "data_paths": {
                "eaglei_data_dir": "eagle-idatasets",
                "outage_data_dir": "outage_data",
                "weather_data_dir": "weather_data",
                "merged_data_dir": "merged_data",
                "results_dir": "results",
                "misc_dir": "misc"
            },

            "file_patterns": {
                "weather_file_pattern": "weather_{state}_{county}_{start}_{end}.parquet",
                "cleaned_weather_file_pattern": "cleaned_weather_{state}_{county}_{start}_{end}.nc",
                "outage_file_pattern": "eaglei_outages_{year}.csv",
                "cleaned_outage_file_pattern": "cleaned_outage_{state}_{county}_{start}_{end}.parquet",
                "merged_file_pattern": "merged_data_{state}_{county}_{start}_{end}.nc",
                "merged_result_file_pattern": "Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.csv"
            }
        }
    
    def _load_config_file(self, config_file: str) -> None:
        """Load and merge configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._merge_configs(self.config, file_config)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {config_file}: {e}")
            raise
    
    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> None:
        """Recursively merge override configuration into base."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base:
                ConfigManager._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command-line arguments."""
        if args.state:
            self.config["processing_parameters"]["state"] = args.state
        if args.county:
            self.config["processing_parameters"]["county"] = args.county
        if args.start_year:
            self.config["processing_parameters"]["start_year"] = args.start_year
        if args.end_year:
            self.config["processing_parameters"]["end_year"] = args.end_year
        # if args.threshold is not None:
        #     self.config["processing_parameters"]["outage_threshold_percentage"] = args.threshold
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation (e.g., 'processing_parameters.state')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return self.config
    
    def get_processing_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters from configuration."""
        return self.get("processing_parameters", {})