# Jmoji

日本語テキストから絵文字列を生成する知識蒸留プロジェクトである。
現在の研究内容は `docs/research_overview.md`、実装手順は `docs/implemention_guide.md`、評価方法は `docs/evaluation.md`、進捗は `docs/status.md` をSoTとする。
教師モデル移行の経緯は `docs/details/teacher_model_migration.md` を参照し、このファイルへ再録しない。

## Environment

Python 3.12とuvは`.mise.toml`で管理する。

```bash
mise install
UV_CACHE_DIR=.uv-cache uv sync
```

依存関係は`uv.lock`へ記録し、プロジェクトコードは`uv run`経由で実行する。
APIキーはgitignoredの`.env`に置き、ログ、テストfixture、コマンドライン、コミットへ含めない。

## Validation

変更範囲に応じて次を実行する。

```bash
uv run pytest tests/
uv run ruff check src/ scripts/ tests/
uv run mypy src/ scripts/
uv run pre-commit run --all-files
```

テストを追加するときは、既存の`tests/`構成とfixtureを優先して再利用する。
ノートブックはjupytextで同期されるため、対応する`notebooks/*.py`を編集し、pre-commitで`.ipynb`を再生成する。

## Sources of truth and synchronization

- 実行例と利用者向けセットアップは`README.md`をSoTとし、CLI引数を変えたら同時に更新する。
- 生成、学習、評価の既定値は`configs/default.yaml`をSoTとし、コードや文書へ同じ値を複製しない。
- データセット生成の設計とseed運用は`docs/details/datasets/`を読み、過去の会話や古い実験値から推測しない。
- 実験結果は`outputs/experiments/<dataset_version>_<experiment_type>_<YYYYMMDD>/`へ記録し、設定、学習ログ、評価指標、予測サンプル、要約を同じ実験単位で管理する。
- モデルcheckpointと一時ログはgit管理対象外のまま保ち、追跡対象へ変更しない。
- 進捗や次の実験を変更したら`docs/status.md`を更新し、このファイルへタスクリストを複製しない。

## Operational constraints

大規模なデータセット生成、外部LLM評価、Hugging Faceへのupload、GPU学習は、費用、外部書込み、または長時間の資源占有を伴う。
これらはユーザーが明示的に依頼した場合だけ実行し、対象、出力先、見込まれる副作用を実行前に示す。
絵文字の抽出、正規化、評価指標を変える場合は既存データとの互換性を確認し、該当する研究・評価文書も更新する。
