import json
from pathlib import Path
from typing import Any, Dict, Optional

from platformdirs import user_config_dir


class SettingsManager:
    """
    Manages application-wide settings.
    """
    def __init__(self, app_name: str, app_author: str, config_file: Optional[Path] = None):
        self.app_name = app_name
        if config_file:
            self.config_file = config_file
            self.config_dir = config_file.parent
        else:
            self.config_dir = Path(user_config_dir(app_name, app_author))
            self.config_file = self.config_dir / "config.json"
        self.settings: Dict[str, Any] = {}
        self.load_settings()

    def load_settings(self):
        """Loads settings from the config file."""
        if self.config_file.exists():
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.settings = json.load(f)
        else:
            # Default settings
            self.settings = {
                "config_version": "0.0.1",
                "ui": {
                    "theme": "dark"
                },
                "node_search_paths": []
            }
            self.save_settings()

    def save_settings(self):
        """Saves the current settings to the config file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a setting value using a dot-separated key.
        e.g., get('ui.theme')
        """
        keys = key.split('.')
        value = self.settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        Sets a setting value using a dot-separated key.
        e.g., set('ui.theme', 'light')
        """
        keys = key.split('.')
        d = self.settings
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self.save_settings()

# グローバルインスタンスは使用側で初期化する
settings_manager: Optional[SettingsManager] = None


def init_settings(config_file: Optional[Path] = None):
    """設定マネージャーを初期化"""
    global settings_manager
    settings_manager = SettingsManager(
        app_name="NodeEditor",
        app_author="NodeEditor",
        config_file=config_file
    )
    return settings_manager


def get_setting(key: str, default: Any = None) -> Any:
    """グローバル設定から値を取得"""
    if settings_manager is None:
        return default
    return settings_manager.get(key, default)
