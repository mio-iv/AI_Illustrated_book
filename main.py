#!/usr/bin/env python3
"""
お子さま専用おしゃべり図鑑

散歩中に見つけた花や虫の写真をアップロードすると、AIがその名前を特定し、
「その子の年齢が理解できる言葉」で解説文を生成します。

使い方:
    python main.py
"""

import base64
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ── モデル設定 ──────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=1024,
)

# ── 年齢別システムプロンプトテンプレート ─────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["age", "language_level", "example_style"],
    template="""あなたは子どもに自然のふしぎを教えてくれる優しい先生です。

{age}歳の子どもに、{language_level}説明してください。

説明のスタイル: {example_style}

必ず以下の内容を含めてください:
1. 名前（なんという生き物・植物か）
2. 見た目の特徴
3. 面白いポイントや豆知識
4. 子どもが「もっと知りたい！」と思える一言

会話口調で、子どもに直接語りかけるように書いてください。""",
)


def get_age_params(age: int) -> dict:
    """年齢に合わせたプロンプトパラメータを返す"""
    if age <= 3:
        return {
            "language_level": "ひらがなだけのとてもかんたんな言葉で、短く",
            "example_style": "「これはちょうちょだよ！きれいだね！」のように親しみやすく短く",
        }
    elif age <= 5:
        return {
            "language_level": "かんたんな言葉（ひらがな・カタカナ中心）で",
            "example_style": "「モンシロチョウっていうちょうちょだよ。白い羽がきれいだね」のように",
        }
    elif age <= 8:
        return {
            "language_level": "小学校低学年にわかる言葉で",
            "example_style": "少し詳しく、でも難しい漢字は使わずに",
        }
    else:
        return {
            "language_level": "小学校高学年にわかる言葉で",
            "example_style": "専門的な名前や仕組みも少し含めて",
        }


def encode_image(image_path: str) -> tuple[str, str]:
    """画像をbase64エンコードし、(data, media_type) を返す"""
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def analyze_image(image_path: str, age: int) -> str:
    """
    画像を解析し、指定年齢向けの解説文を生成する。

    LangChainの構成:
        SystemMessage (PromptTemplate で年齢別に生成)
            └─ HumanMessage (base64画像 + テキスト)
                └─ ChatAnthropic (claude-sonnet-4-6 Vision)
    """
    age_params = get_age_params(age)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(age=age, **age_params)

    image_data, media_type = encode_image(image_path)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "この写真に写っている生き物や植物を教えてください！",
                },
            ]
        ),
    ]

    response = llm.invoke(messages)
    return response.content


# ── Gradio UI ───────────────────────────────────────────────────────────────
def gradio_analyze(image_path: str | None, age: int) -> str:
    """Gradio から呼び出されるハンドラ"""
    if image_path is None:
        return "写真をアップロードしてください。"
    return analyze_image(image_path, age)


with gr.Blocks(title="ものしりずかん") as demo:
    gr.Markdown("# ものしり ずかん")
    gr.Markdown("おさんぽで見つけたいきものの写真をアップすると、AIがおしえてくれるよ！見つけた画像をアップしてね！")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="写真をアップロード")
            age_slider = gr.Slider(
                minimum=1, maximum=18, step=1, value=4, label="お子さまの年齢"
            )
            submit_btn = gr.Button("教えて！", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="AIの解説", lines=12)

    submit_btn.click(
        fn=gradio_analyze,
        inputs=[image_input, age_slider],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
