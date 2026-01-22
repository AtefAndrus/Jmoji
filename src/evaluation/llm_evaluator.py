"""LLM-based multi-evaluator with persona support.

Claude Code subagentsを使って、異なるペルソナで絵文字予測モデルを評価する。
"""

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PersonaConfig:
    """ペルソナ設定."""

    id: str
    name: str
    description: str
    evaluation_priority: list[str]
    bias: str


def get_personas() -> dict[str, PersonaConfig]:
    """5つのペルソナ設定を返す."""
    return {
        "persona_1": PersonaConfig(
            id="persona_1_young_sns_user",
            name="若者SNSユーザー（18-25歳）",
            description="Z世代のSNSヘビーユーザー。Twitter、Instagram、TikTokを日常的に使用し、絵文字を頻繁に活用する。流行の絵文字や使い方に敏感。",
            evaluation_priority=["naturalness", "semantic"],
            bias="自然さと流行への適合を重視。同じ絵文字の繰り返しは不自然と感じやすい。",
        ),
        "persona_2": PersonaConfig(
            id="persona_2_business_professional",
            name="ビジネスパーソン（35-45歳）",
            description="中堅企業の会社員。SNSは控えめに使用し、絵文字は誤解を避けるため慎重に選ぶ。公式感や適切さを重視。",
            evaluation_priority=["misleading", "semantic"],
            bias="誤解を招く表現に厳格。感情表現は控えめな方が良いと考える傾向。",
        ),
        "persona_3": PersonaConfig(
            id="persona_3_elderly_user",
            name="シニアユーザー（60-70歳）",
            description="LINEを中心に家族や友人とコミュニケーション。絵文字は基本的なもの（😊❤️👍等）を好み、複雑な絵文字は避ける。",
            evaluation_priority=["semantic", "naturalness"],
            bias="シンプルでわかりやすい絵文字を好む。多すぎる絵文字や専門的な絵文字は理解しにくいと感じる。",
        ),
        "persona_4": PersonaConfig(
            id="persona_4_creative",
            name="クリエイター（25-35歳）",
            description="デザイナーやライター。表現の豊かさや多様性を重視。絵文字の組み合わせで感情やニュアンスを細かく表現することを好む。",
            evaluation_priority=["semantic", "naturalness"],
            bias="表現の多様性と創造性を高く評価。定番絵文字ばかりだと物足りなく感じる。",
        ),
        "persona_5": PersonaConfig(
            id="persona_5_researcher",
            name="研究者（30-40歳）",
            description="言語学やAI分野の研究者。客観性と精度を重視し、教師データとの一致度を中心に評価。感情的バイアスを排除し分析的。",
            evaluation_priority=["semantic", "misleading"],
            bias="教師データ（gold）との一致度を重視。主観的な「自然さ」よりも客観的な「正確さ」を優先。",
        ),
    }


def get_few_shot_examples() -> list[dict[str, Any]]:
    """評価のFew-shot examplesを返す."""
    return [
        {
            "sample_id": 0,
            "text": "大会で優勝して本当に嬉しい！",
            "gold": "🏆 🎉 😊 👏 ✨",
            "model_a": "🎉 👏 😊 😊 😊",
            "model_b": "🏆 🎉 🇯🇵 😊 👏",
            "evaluation": {
                "gold": {
                    "semantic": 4,
                    "naturalness": 4,
                    "misleading": False,
                },
                "model_a": {
                    "semantic": 3,
                    "naturalness": 2,
                    "misleading": False,
                },
                "model_b": {
                    "semantic": 4,
                    "naturalness": 3,
                    "misleading": False,
                },
                "preference": "B（top50）",
                "comment": "モデルBは🏆を含み優勝の文脈を捉えている。モデルAは😊の繰り返しが不自然。goldは完璧。",
            },
        },
        {
            "sample_id": 0,
            "text": "自性分別ってのは、尋と伺のことで、分別の一種ってことね。",
            "gold": "🤔 🧠 📖 ✨ 🔍",
            "model_a": "😊 😊 😊 😊 😊",
            "model_b": "📖 😊 🎉 💼 🇯🇵",
            "evaluation": {
                "gold": {
                    "semantic": 4,
                    "naturalness": 3,
                    "misleading": False,
                },
                "model_a": {
                    "semantic": 0,
                    "naturalness": 0,
                    "misleading": True,
                },
                "model_b": {
                    "semantic": 2,
                    "naturalness": 1,
                    "misleading": False,
                },
                "preference": "B（top50）",
                "comment": "モデルAは完全なmode collapse（😊のみ連発）で誤解を招く。モデルBは📖で仏教用語の文脈を部分的に捉えているが、🎉は不適切。goldは思考系絵文字（🤔🧠🔍）で概念を適切に表現。",
            },
        },
    ]


class LLMEvaluator:
    """LLMベースの絵文字予測評価器（ペルソナサポート）."""

    def __init__(self, persona: PersonaConfig):
        """初期化.

        Args:
            persona: ペルソナ設定
        """
        self.persona = persona

    def build_prompt(self, samples: list[dict[str, Any]]) -> str:
        """評価プロンプトを生成.

        Args:
            samples: 評価サンプルリスト（text, gold, pred_focal_top50, pred_top50を含む）

        Returns:
            評価プロンプト
        """
        # Few-shot examples
        examples = get_few_shot_examples()
        examples_text = ""
        for i, ex in enumerate(examples, 1):
            eval_data = ex["evaluation"]
            examples_text += f"""
## サンプル例 {i}

**入力文**: {ex["text"]}

**教師（Gold）**: {ex["gold"]}
**モデルA（focal_top50）**: {ex["model_a"]}
**モデルB（top50）**: {ex["model_b"]}

**評価**:
```json
{{
  "sample_id": {ex["sample_id"]},
  "gold": {json.dumps(eval_data["gold"], ensure_ascii=False)},
  "model_a": {json.dumps(eval_data["model_a"], ensure_ascii=False)},
  "model_b": {json.dumps(eval_data["model_b"], ensure_ascii=False)},
  "preference": "{eval_data["preference"]}",
  "comment": "{eval_data["comment"]}"
}}
```
"""

        # サンプルデータ
        samples_text = ""
        for i, sample in enumerate(samples, 1):
            samples_text += f"""
## サンプル {i}

**入力文**: {sample["text"]}

**教師（Gold）**: {sample["gold"]}
**モデルA（focal_top50）**: {sample["pred_focal_top50"]}
**モデルB（top50）**: {sample["pred_top50"]}
"""

        # プロンプト生成
        prompt = f"""# 役割とペルソナ

あなたは{self.persona.name}です。

{self.persona.description}

あなたの評価観点: {", ".join(self.persona.evaluation_priority)}
評価バイアス: {self.persona.bias}

# タスク定義

日本語テキストに対する絵文字予測モデルの出力を評価してください。

評価対象:
- **教師（Gold）**: Qwen3-235B-A22Bが生成した絵文字列
- **モデルA（focal_top50）**: T5学生モデル1（Focal Loss適用）
- **モデルB（top50）**: T5学生モデル2（標準学習）

# 評価基準

以下の基準で各モデル出力を評価してください:

## 1. 意味的一致度 (semantic) [0-4]

テキストの意味・感情を絵文字がどの程度表現しているか

- **0**: ほとんど関係がない
- **1**: 部分的に関連しているが不自然
- **2**: 一応意味は通るが不十分
- **3**: 概ね妥当
- **4**: 非常に妥当で自然

## 2. 自然さ (naturalness) [0-4]

日本のSNS（Twitter/X、Instagram等）で見かけそうな使い方か

- **0**: 全く不自然、違和感がある
- **1**: やや不自然
- **2**: 普通
- **3**: 自然
- **4**: 非常に自然、よく見かける使い方

## 3. 誤解の可能性 (misleading) [true/false]

絵文字の選択が、元の文の意図と**逆の印象**を与えそうか

- **true**: 誤解を招く可能性がある（例: ネガティブな文に😊🎉など）
- **false**: 誤解を招かない

## 4. モデル選好 (preference) [A/B/同等]

2つの学生モデル（A vs B）のどちらの出力が良いか

- **"A（focal_top50）"**: モデルAの方が良い
- **"B（top50）"**: モデルBの方が良い
- **"同等"**: どちらも同程度、または両方とも不十分

# Few-shot Examples
{examples_text}

# 出力形式

以下のJSON配列形式で、**全サンプルの評価を一度に**出力してください:

```json
[
  {{
    "sample_id": 1,
    "gold": {{"semantic": 3, "naturalness": 3, "misleading": false}},
    "model_a": {{"semantic": 2, "naturalness": 1, "misleading": false}},
    "model_b": {{"semantic": 1, "naturalness": 2, "misleading": true}},
    "preference": "A（focal_top50）",
    "comment": "評価理由を日本語で簡潔に"
  }},
  {{
    "sample_id": 2,
    ...
  }}
]
```

**重要な注意事項**:
- 必ず有効なJSON配列形式で出力してください
- sample_idは1から始まる連番
- preferenceは必ず "A（focal_top50）"、"B（top50）"、"同等" のいずれかを使用
- commentはあなたのペルソナの観点から評価理由を簡潔に日本語で記述

# 評価対象サンプル
{samples_text}

**それでは、{self.persona.name}の視点で、上記{len(samples)}件のサンプルを評価し、JSON配列形式で出力してください。**
"""
        return prompt

    def parse_response(self, response: str) -> list[dict[str, Any]]:
        """LLMのJSON応答を解析.

        Args:
            response: LLMの応答テキスト

        Returns:
            評価結果のリスト

        Raises:
            ValueError: JSON解析エラー
        """
        # JSON codeblockを抽出
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # codeblockなしの場合、全体をJSON としてパース
            json_str = response

        try:
            evaluations = json.loads(json_str)
        except json.JSONDecodeError as e:
            # エラー回復: 一般的な問題を修正
            json_str = self._fix_common_json_errors(json_str)
            try:
                evaluations = json.loads(json_str)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse JSON response: {e}\nResponse: {response[:500]}"
                ) from e

        # リストでない場合、リストにラップ
        if not isinstance(evaluations, list):
            evaluations = [evaluations]

        # 各評価をバリデーション
        for eval_item in evaluations:
            self.validate_evaluation(eval_item)

        return evaluations

    def _fix_common_json_errors(self, json_str: str) -> str:
        """一般的なJSON エラーを修正.

        Args:
            json_str: JSON文字列

        Returns:
            修正されたJSON文字列
        """
        # trailing commaを削除
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # single quoteをdouble quoteに変換
        # （注意: これは完全ではないが、簡単なケースには有効）
        # json_str = json_str.replace("'", '"')

        return json_str

    def validate_evaluation(self, eval_dict: dict[str, Any]) -> None:
        """評価結果のスキーマをバリデート.

        Args:
            eval_dict: 評価結果の辞書

        Raises:
            ValueError: スキーマ違反
        """
        required_fields = ["sample_id", "gold", "model_a", "model_b", "preference"]
        for field in required_fields:
            if field not in eval_dict:
                raise ValueError(f"Missing required field: {field}")

        # sample_idチェック
        if not isinstance(eval_dict["sample_id"], int) or eval_dict["sample_id"] < 1:
            raise ValueError(
                f"sample_id must be positive integer, got {eval_dict['sample_id']}"
            )

        # gold, model_a, model_b のスキーマチェック
        for model in ["gold", "model_a", "model_b"]:
            model_data = eval_dict[model]
            if not isinstance(model_data, dict):
                raise ValueError(f"{model} must be a dict")

            required_model_fields = ["semantic", "naturalness", "misleading"]
            for field in required_model_fields:
                if field not in model_data:
                    raise ValueError(f"Missing field in {model}: {field}")

            # semanticとnaturalnessは0-4
            for field in ["semantic", "naturalness"]:
                value = model_data[field]
                if not isinstance(value, int) or not 0 <= value <= 4:
                    raise ValueError(
                        f"{model}.{field} must be integer 0-4, got {value}"
                    )

            # misleadingはboolean
            if not isinstance(model_data["misleading"], bool):
                raise ValueError(
                    f"{model}.misleading must be boolean, got {model_data['misleading']}"
                )

        # preferenceチェック
        valid_preferences = ["A（focal_top50）", "B（top50）", "同等"]
        if eval_dict["preference"] not in valid_preferences:
            raise ValueError(
                f"preference must be one of {valid_preferences}, got {eval_dict['preference']}"
            )
