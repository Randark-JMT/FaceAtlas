"""配置管理模块 - 资源文件固定，数据库连接可配置（PostgreSQL）"""

import os
import re
import sys
import json
from typing import Optional


# ---- 常量 ----

APP_NAME = "FaceAtlas"


# ---- 辅助函数 ----

def _get_appdata_base() -> str:
    """获取 AppData 下的应用目录（Windows: %APPDATA%/FaceAtlas）"""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    path = os.path.join(base, APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _get_resource_base() -> str:
    """获取资源文件基目录（模型等只读资源）

    打包后：PyInstaller 的 _MEIPASS 临时目录
    源代码运行：项目目录
    """
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        import __main__
        if hasattr(__main__, "__file__"):
            return os.path.dirname(os.path.abspath(__main__.__file__))
        return os.getcwd()


class Config:
    """配置管理器"""

    LOG_FILENAME = "FaceAtlas.log"

    def __init__(self):
        # AppData 目录（固定）
        self._appdata_dir = _get_appdata_base()

        # 配置文件放在 AppData 下
        self._config_file = os.path.join(self._appdata_dir, "config.json")

        # 资源文件基目录（固定）
        self._resource_base = _get_resource_base()

        # PostgreSQL 连接配置
        self._pg_host: str = "localhost"
        self._pg_port: int = 5432
        self._pg_user: str = "postgres"
        self._pg_password: str = ""
        self._pg_database: str = "faceatlas"
        self._pg_configured: bool = False
        self._detection_input_max_side: int = 0

        self._load_config()

    # ================================================
    # 配置文件读写
    # ================================================

    def _load_config(self):
        """从 AppData/config.json 加载配置"""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._pg_host = str(data.get("pg_host", "localhost"))
                    self._pg_port = int(data.get("pg_port", 5432))
                    self._pg_user = str(data.get("pg_user", "postgres"))
                    self._pg_password = str(data.get("pg_password", ""))
                    self._pg_database = str(data.get("pg_database", "faceatlas"))
                    self._pg_configured = bool(data.get("pg_configured", False))
                    # 检测输入最长边上限，0=不限制；640/1280 可提高 CUDA 吞吐与利用率
                    self._detection_input_max_side = max(0, int(data.get("detection_input_max_side", 0)))
            except Exception as e:
                print(f"警告：加载配置文件失败: {e}")

    def _save_config(self):
        """保存配置到 AppData/config.json"""
        try:
            data = {
                "pg_host": self._pg_host,
                "pg_port": self._pg_port,
                "pg_user": self._pg_user,
                "pg_password": self._pg_password,
                "pg_database": self._pg_database,
                "pg_configured": self._pg_configured,
                "detection_input_max_side": self._detection_input_max_side,
            }
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告：保存配置文件失败: {e}")

    # ================================================
    # 数据目录（兼容保留）
    # ================================================

    @property
    def default_data_dir(self) -> str:
        """兼容旧代码：统一使用 AppData"""
        return self._appdata_dir

    @property
    def data_dir(self) -> str:
        """兼容旧代码：统一使用 AppData"""
        os.makedirs(self._appdata_dir, exist_ok=True)
        return self._appdata_dir

    @property
    def is_data_dir_configured(self) -> bool:
        """兼容旧代码：总是 True"""
        return True

    def set_data_dir(self, path: Optional[str]):
        """兼容旧接口：保留空实现"""
        _ = path

    # ---- 兼容旧属性名 ----

    @property
    def cache_dir(self) -> str:
        """兼容旧代码：等同于 data_dir"""
        return self.data_dir

    @cache_dir.setter
    def cache_dir(self, path: str):
        self.set_data_dir(path)

    # ================================================
    # PostgreSQL 配置
    # ================================================

    @property
    def is_pg_configured(self) -> bool:
        return self._pg_configured

    @property
    def pg_host(self) -> str:
        return self._pg_host

    @property
    def pg_port(self) -> int:
        return self._pg_port

    @property
    def pg_user(self) -> str:
        return self._pg_user

    @property
    def pg_password(self) -> str:
        return self._pg_password

    @property
    def pg_database(self) -> str:
        return self._pg_database

    def set_pg_connection(self, host: str, port: int, user: str, password: str, database: str):
        self._pg_host = host.strip() or "localhost"
        self._pg_port = int(port)
        self._pg_user = user.strip() or "postgres"
        self._pg_password = password
        self._pg_database = database.strip()
        self._pg_configured = True
        self._save_config()

    @property
    def database_path(self) -> str:
        """兼容旧日志字段名：返回 PostgreSQL 连接标识"""
        return f"postgresql://{self._pg_user}@{self._pg_host}:{self._pg_port}/{self._pg_database}"

    @property
    def database_display(self) -> str:
        return f"{self._pg_host}:{self._pg_port}/{self._pg_database}"

    @property
    def log_path(self) -> str:
        """日志文件路径（按数据库名区分）"""
        logs_dir = os.path.join(self._appdata_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        db_part = self._sanitize_name(self._pg_database or "default")
        return os.path.join(logs_dir, f"FaceAtlas-{db_part}.log")

    @property
    def thumb_cache_dir(self) -> str:
        """缩略图缓存目录（按数据库名隔离）"""
        cache_root = os.path.join(self._appdata_dir, "thumb_cache")
        os.makedirs(cache_root, exist_ok=True)
        db_part = self._sanitize_name(self._pg_database or "default")
        cache_dir = os.path.join(cache_root, db_part)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @property
    def detection_input_max_side(self) -> int:
        """检测时输入图像最长边上限（0=不限制）。设为 640 或 1280 可提高 CUDA 利用率与吞吐。"""
        return getattr(self, "_detection_input_max_side", 0)

    # ================================================
    # 资源文件路径
    # ================================================

    def get_resource_path(self, relative_path: str) -> str:
        """获取只读资源文件路径（模型等）

        打包后从 _MEIPASS 读取，源代码运行从项目目录读取。
        """
        return os.path.join(self._resource_base, relative_path)

    @property
    def executable_dir(self) -> str:
        """可执行文件所在目录"""
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return self._resource_base

    # ================================================
    # 数据文件检查
    # ================================================

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """将任意名称转换为安全文件名片段"""
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        return cleaned[:64] if cleaned else "default"


# ---- 全局单例 ----

_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config
