"""PostgreSQL 连接与数据库选择对话框。"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.config import Config
from core.database import DatabaseManager
from ui import APP_NAME


class PgConnectDialog(QDialog):
    """启动时选择 PostgreSQL 连接与项目数据库。"""

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._result: dict | None = None

        self.setWindowTitle(APP_NAME)
        self.setFixedWidth(520)

        self._build_ui()
        self._apply_style()
        self._load_from_config()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        title = QLabel("连接 PostgreSQL 并选择项目数据库")
        title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        root.addWidget(title)

        desc = QLabel(
            "一个项目对应一个独立数据库。\n"
            "可从下拉列表选择数据库，也可手动输入；若不存在会自动创建。"
        )
        desc.setStyleSheet("color: #aaa;")
        desc.setWordWrap(True)
        root.addWidget(desc)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._host_edit = QLineEdit()
        self._port_edit = QLineEdit()
        self._user_edit = QLineEdit()
        self._pass_edit = QLineEdit()
        self._pass_edit.setEchoMode(QLineEdit.EchoMode.Password)

        self._db_combo = QComboBox()
        self._db_combo.setEditable(True)

        form.addRow("Host:", self._host_edit)
        form.addRow("Port:", self._port_edit)
        form.addRow("User:", self._user_edit)
        form.addRow("Password:", self._pass_edit)
        form.addRow("Database:", self._db_combo)

        root.addLayout(form)

        ops = QHBoxLayout()
        self._refresh_btn = QPushButton("刷新数据库列表")
        self._refresh_btn.clicked.connect(self._refresh_databases)
        ops.addWidget(self._refresh_btn)
        ops.addStretch()
        root.addLayout(ops)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        root.addWidget(self._status)

        btns = QHBoxLayout()
        btns.addStretch()
        self._ok_btn = QPushButton("连接")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_ok)
        btns.addWidget(self._ok_btn)
        root.addLayout(btns)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog { background-color: #1e1e1e; color: #d4d4d4; }
            QLineEdit, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QComboBox QAbstractItemView {
                background-color: #252526;
                border: 1px solid #555;
                color: #d4d4d4;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                border-radius: 3px;
                padding: 5px 14px;
                color: #fff;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5689; }
            """
        )

    def _load_from_config(self):
        """从配置加载表单，不自动连接数据库，避免启动卡顿（远程/无本地库时）。"""
        self._host_edit.setText(self.config.pg_host)
        self._port_edit.setText(str(self.config.pg_port))
        self._user_edit.setText(self.config.pg_user)
        self._pass_edit.setText(self.config.pg_password)
        self._db_combo.setCurrentText(self.config.pg_database)
        # 不自动调用 _refresh_databases()，需用户手动点击「刷新数据库列表」

    def _set_status(self, text: str, ok: bool):
        self._status.setVisible(True)
        if ok:
            self._status.setStyleSheet(
                "font-size: 9pt; padding: 6px; color: #4ec9b0; background: #1a3a2a; border-radius: 3px;"
            )
        else:
            self._status.setStyleSheet(
                "font-size: 9pt; padding: 6px; color: #f48771; background: #3a1a1a; border-radius: 3px;"
            )
        self._status.setText(text)

    def _read_inputs(self) -> tuple[str, int, str, str, str] | None:
        host = self._host_edit.text().strip()
        port_text = self._port_edit.text().strip()
        user = self._user_edit.text().strip()
        password = self._pass_edit.text()
        database = self._db_combo.currentText().strip()

        if not host or not port_text or not user or not database:
            QMessageBox.warning(self, "参数不完整", "请填写 Host / Port / User / Database。")
            return None
        try:
            port = int(port_text)
        except ValueError:
            QMessageBox.warning(self, "端口错误", "Port 必须是数字。")
            return None
        return host, port, user, password, database

    def _refresh_databases(self):
        values = self._read_inputs()
        if values is None:
            return

        host, port, user, password, current_db = values
        try:
            names = DatabaseManager.list_databases(host, port, user, password)
            self._db_combo.blockSignals(True)
            self._db_combo.clear()
            self._db_combo.addItems(names)
            self._db_combo.setCurrentText(current_db)
            self._db_combo.blockSignals(False)
            self._set_status(f"已连接服务器，发现 {len(names)} 个数据库。", True)
        except Exception as e:
            self._set_status(f"连接服务器失败：{e}", False)

    def _on_ok(self):
        values = self._read_inputs()
        if values is None:
            return

        host, port, user, password, database = values
        try:
            created = DatabaseManager.ensure_database_exists(host, port, user, password, database)
            valid, msg = DatabaseManager.validate_database_schema(host, port, user, password, database)
            if not valid:
                QMessageBox.critical(self, "拒绝连接", msg)
                self._set_status(msg, False)
                return
            if created:
                self._set_status(f"数据库 {database} 不存在，已自动创建。{msg}", True)
            else:
                self._set_status(msg, True)
        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"无法连接或创建数据库：\n{e}")
            self._set_status(f"连接失败：{e}", False)
            return

        self._result = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self.accept()

    def get_result(self) -> dict | None:
        return self._result


def show_pg_connect_dialog(config: Config, parent=None) -> bool:
    dlg = PgConnectDialog(config, parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False
    result = dlg.get_result()
    if not result:
        return False
    config.set_pg_connection(
        result["host"],
        result["port"],
        result["user"],
        result["password"],
        result["database"],
    )
    return True
