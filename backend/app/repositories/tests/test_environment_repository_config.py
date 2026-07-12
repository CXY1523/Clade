from pathlib import Path

from app.models.config import UIConfig
from app.repositories.environment_repository import EnvironmentRepository


def test_save_ui_config_replaces_file_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"autosave_enabled": false}', encoding="utf-8")
    repo = EnvironmentRepository()
    replacements: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def tracking_replace(source: Path, target: Path) -> Path:
        replacements.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", tracking_replace)

    repo.save_ui_config(path, UIConfig(autosave_enabled=True))

    assert UIConfig.model_validate_json(path.read_text(encoding="utf-8")).autosave_enabled is True
    assert replacements == [(path.with_suffix(".json.tmp"), path)]
    assert not path.with_suffix(".json.tmp").exists()
