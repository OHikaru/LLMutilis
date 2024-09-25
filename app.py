import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import csv
from PyPDF2 import PdfReader
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_MODELS = [
    "models/gemini-1.5-pro-002",
    "models/gemini-pro-vision",
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-pro-exp-0801",
    "models/gemini-1.5-pro-exp-0827",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-001-tuning",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-flash-exp-0827",
    "models/gemini-1.5-flash-8b-exp-0827",
    "models/gemini-1.5-flash-8b-exp-0924",
    "models/gemini-1.5-flash-002",
]

OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12"
]

IMAGE_META_PROMPT = """
<ocr_prompt>
  <instruction>
    この日本語手書き画像に対して最高精度のOCRを実行し、以下の構造に厳密に従って解析する。\n\nOCRした文字のみを最終的に出力する。\n\n<output_structure>の情報は解析のためにのみ使用し、認識した文字以外は一切出力してはいけない。\n\n余分な説明や構造化は避け、検出された情報のみを簡潔に記載する。
  </instruction>
  <output_structure>
    文書情報:
    向き: [縦書き/横書き]
    表数: [数値]
    全体信頼度: [0-100]
    用紙サイズ: [A4/B5/その他]
    記入方法: [ペン/鉛筆/その他]
    インク色: [黒/青/赤/その他]

    表[N]: (Nは表番号)
    構造:
      行数: [数値]
      列数: [数値]
      ヘッダー行: [行番号/なし]
    罫線:
      種類: [実線/点線/なし]
      状態: [明瞭/かすれ/途切れ]
    セル[行,列]:
      内容: [テキスト]
      種類: [自由記載/選択肢/ヘッダー]
      信頼度: [0-100]
      座標: [左上x,左上y,右下x,右下y]
      文字:
        [文字]: 種類[漢字/ひらがな/カタカナ/記号/英字/数字], 信頼度[0-100], 書体[楷書/行書/草書/ゴシック体/その他]
        (漢字の場合) 読み: 音[音読み], 訓[訓読み]
      特記事項: [修正痕/上書き/取り消し線/その他]

    自由記載:
    項目[N]: (Nは項目番号)
      内容: [テキスト]
      位置: 表[数値], 行[数値], 列[数値]
      信頼度: [0-100]
      言語: [日本語/英語/その他]
      文字詳細: (上記のセル内の文字と同様の形式)

    選択肢:
    項目[N]: (Nは項目番号)
      選択肢: [テキスト]
      位置: 表[数値], 行[数値], 列[数値]
      マーク:
        種類: [円/チェック/塗りつぶし/その他]
        完全性: [0-100]
        サイズ: [0-100]
        重なり: [0-100]
      信頼度: [0-100]

    特殊項目:
    日付[N]: (Nは項目番号)
      内容: [テキスト]
      形式: [和暦/西暦]
      位置: 表[数値], 行[数値], 列[数値]
      信頼度: [0-100]
    数値[N]: (Nは項目番号)
      内容: [テキスト]
      種類: [漢数字/アラビア数字]
      位置: 表[数値], 行[数値], 列[数値]
      信頼度: [0-100]
    固有名詞[N]: (Nは項目番号)
      内容: [テキスト]
      分類: [人名/地名/組織名/その他]
      位置: 表[数値], 行[数値], 列[数値]
      信頼度: [0-100]

    署名:
    位置: [文書上部/文書下部/その他]
    種類: [印鑑/サイン/その他]
    文字: [判読可能な文字]
    信頼度: [0-100]

    追加情報:
    文書品質: [良好/かすれあり/しみあり/折り目あり/その他]
    筆跡一貫性: [高/中/低]
    特殊記号: [数式/矢印/図形/その他]
    余白メモ: [あり/なし]
    (ありの場合) 内容: [テキスト], 位置: [上部/下部/左側/右側]
  </output_structure>
  <processing_instructions>
    <instruction>ページが見開きの場合、各ページごとに個別にOCRを行い、ページごとに結果をまとめる。</instruction>
    <instruction>文字の読みが不明な場合は「不明」と記載してください。</instruction>
    <instruction>表が複数ある場合、それぞれを個別に解析し、関連性があれば注記してください。</instruction>
    <instruction>くずし字や特殊な書体は可能な限り解読し、解読困難な場合はその旨を記載してください。</instruction>
    <instruction>選択肢の判断は、マークの完全性、サイズ、重なりを総合的に評価して行ってください。</instruction>
    <instruction>筆圧の変化や筆の運びを分析し、筆跡の一貫性を評価してください。</instruction>
    <instruction>表の罫線や枠線の状態を詳細に分析し、手書きか印刷かを判断してください。</instruction>
    <instruction>文書全体の品質（かすれ、しみ、折り目など）を評価し、OCR精度への影響を考慮してください。</instruction>
    <instruction>日本語と英語が混在している場合、それぞれを適切に識別し処理してください。</instruction>
    <instruction>数式や特殊記号は可能な限り正確に認識し、その意味を解釈してください。</instruction>
    <instruction>修正痕、上書き、取り消し線などがある場合、元の文字と修正後の文字の両方を記録してください。</instruction>
    <instruction>署名や印鑑がある場合、その位置と特徴を詳細に記述してください。</instruction>
    <instruction>余白や欄外のメモ、注釈なども見落とさず記録してください。</instruction>
    <instruction>文字の配置や間隔に不自然さがある場合（例：空白が広すぎる）、その旨を記載してください。</instruction>
    <instruction>文書の向きが途中で変わっている場合（例：縦書きと横書きの混在）、その変化を明確に記録してください。</instruction>
  </processing_instructions>
</ocr_prompt>

{user_prompt}

上記の指示に従って、以下の内容を処理してください。

出力結果から文字のみ抜き出して、日本語として意味が分かるように再構成して表示すること。：

{content}
"""

TEXT_META_PROMPT = """
あなたは高度な日本語AIアシスタントです。ユーザーからの質問に対して、以下の指針に従って回答してください：

1. 正確性: 提供する情報は常に正確で最新のものを心がけてください。
2. 簡潔性: 回答は簡潔かつ明瞭であるべきです。不必要な冗長さは避けてください。
3. 丁寧さ: 常に礼儀正しく、敬意を持って対応してください。
4. 柔軟性: ユーザーの質問の意図を理解し、適切に対応してください。
5. 専門性: 専門的な話題に関しては、可能な限り詳細かつ正確な情報を提供してください。
6. 中立性: 意見を求められた場合でも、可能な限り中立的な立場を保ってください。
7. 文化的配慮: 日本の文化や慣習を考慮に入れて回答してください。
8. 補足説明: 必要に応じて、追加の説明や例を提供し、理解を深めるよう努めてください。
9. 制限の認識: 自身の知識や能力の限界を認識し、不確かな情報は提供しないでください。
10. フォローアップ: ユーザーの理解度を確認し、必要に応じて追加の質問を促してください。

{user_prompt}

上記の指針に従って、以下の質問に答えてください：

{content}
"""

temperature=1.0 #temperatureを指定

def setup_gemini_model(api_key, model_name):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def setup_openai_model(api_key):
    return OpenAI(api_key=api_key)

def process_with_gemini(model, content, prompt, is_image):
    try:
        if is_image:
            response = model.generate_content([IMAGE_META_PROMPT, content])
        else:
            response = model.generate_content(TEXT_META_PROMPT + "\n\n" + content)
        return response.text
    except Exception as e:
        st.error(f"Error processing with Gemini: {str(e)}")
        return None

def process_with_openai(client, content, prompt, is_image, model_name):
    try:
        if is_image:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": IMAGE_META_PROMPT},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": content}}]}
                ]
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TEXT_META_PROMPT},
                    {"role": "user", "content": content}
                ]
            )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error processing with OpenAI: {str(e)}")
        return None

def process_content(content, api_key, prompt, model_name, is_image):
    if "gemini" in model_name.lower():
        model = setup_gemini_model(api_key, model_name)
        return process_with_gemini(model, content, prompt, is_image)
    elif "gpt" in model_name.lower():
        client = setup_openai_model(api_key)
        return process_with_openai(client, content, prompt, is_image, model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def clean_csv(csv_str):
    try:
        # 入力文字列をSHIFT-JISでエンコード
        csv_bytes = csv_str.encode('shift_jis')
        
        # バイト列からSHIFT-JISでデコード
        csv_str = csv_bytes.decode('shift_jis')
        
        lines = [line.strip() for line in csv_str.split('\n') if line.strip()]
        csv_reader = csv.reader(lines)
        cleaned_rows = list(csv_reader)

        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(cleaned_rows)

        # StringIOの内容をSHIFT-JISでエンコード
        result_bytes = output.getvalue().encode('shift_jis')
        
        return result_bytes
    except UnicodeEncodeError:
        print("エンコーディングエラー: 入力データをSHIFT-JISに変換できません。")
        return None
    except UnicodeDecodeError:
        print("デコードエラー: 入力データをSHIFT-JISとして解釈できません。")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def main():
    st.set_page_config(page_title="AIドキュメント処理システム", page_icon="📄", layout="wide")
    st.title('AIドキュメント・画像処理 & Q&Aシステム')

    # Initialize session state
    if 'content' not in st.session_state:
        st.session_state.content = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = GEMINI_MODELS[0]

    # Sidebar for configuration
    with st.sidebar:
        st.header("設定")
        api_key_type = st.radio("APIキーの種類を選択してください。", ("Gemini", "OpenAI"))
        api_key = st.text_input(f"{api_key_type} APIキー", type="password", value=st.session_state.api_key)
        
        models = GEMINI_MODELS if api_key_type == "Gemini" else OPENAI_MODELS
        selected_model = st.selectbox("AIモデルを選択してください。\n\n OCRのおすすめはmodels/gemini-1.5-pro-002です", models, index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
        
        st.session_state.api_key = api_key
        st.session_state.selected_model = selected_model

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("入力")
        input_type = st.radio("入力のタイプを選択してください。", ("テキスト", "画像/PDF"))

        if input_type == "テキスト":
            user_input = st.text_area("質問や処理したいテキストを入力してください")
            process_button = st.button('テキストを処理')
            
            if process_button and user_input and st.session_state.api_key:
                with st.spinner('処理中...'):
                    try:
                        result = process_content(user_input, st.session_state.api_key, "", st.session_state.selected_model, is_image=False)
                        if result:
                            st.session_state.results.append({"result": result, "prompt": user_input, "model": st.session_state.selected_model})
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")
        else:
            initial_prompt = st.text_area("追加で指示したい内容があれば入力してください（オプション）")
            uploaded_file = st.file_uploader("画像またはPDFファイルを選択してください。", type=["jpg", "jpeg", "png", "pdf", "tiff", "bmp", "webp"])

            if uploaded_file:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption='アップロードされた画像', use_column_width=True)
                    st.session_state.content = image
                    is_image = True
                elif uploaded_file.type == 'application/pdf':
                    st.session_state.content = extract_text_from_pdf(uploaded_file)
                    st.text(f"PDFの内容（プレビュー）:\n{st.session_state.content[:500]}...")
                    is_image = False
                else:
                    st.error("対応していないファイル形式です。画像またはPDFをアップロードしてください。")
                    st.stop()

                process_button = st.button('ファイルを処理')
                
                if process_button and st.session_state.api_key:
                    with st.spinner('処理中...'):
                        try:
                            result = process_content(st.session_state.content, st.session_state.api_key, initial_prompt, st.session_state.selected_model, is_image=is_image)
                            if result:
                                st.session_state.results.append({"result": result, "prompt": initial_prompt, "model": st.session_state.selected_model})
                        except Exception as e:
                            st.error(f"エラーが発生しました: {str(e)}")

    with col2:
        st.subheader("結果")
        for i, result_data in enumerate(st.session_state.results):
            with st.expander(f"結果 {i+1} (モデル: {result_data['model']})"):
                st.text_area(f"出力", result_data["result"], height=200, key=f"output_{i}")
                
                csv_result = clean_csv(result_data["result"])
                st.download_button(
                    label=f"CSVとしてダウンロード",
                    data=csv_result,
                    file_name=f"result_{i+1}.csv",
                    mime="text/csv",
                    key=f"download_{i}"
                )

                additional_prompt = st.text_area("追加の指示を入力", key=f"additional_prompt_{i}")
                reprocess_button = st.button('再処理', key=f"reprocess_{i}")
                
                if reprocess_button:
                    with st.spinner('再処理中...'):
                        try:
                            new_result = process_content(result_data["result"], st.session_state.api_key, additional_prompt, st.session_state.selected_model, is_image=False)
                            if new_result:
                                st.session_state.results.append({"result": new_result, "prompt": additional_prompt, "model": st.session_state.selected_model})
                        except Exception as e:
                            st.error(f"エラーが発生しました: {str(e)}")

    # 新しい結果を表示するためのコードをループの外に移動
    if st.session_state.results:
        st.subheader("いちばん最新の結果")
        latest_result = st.session_state.results[-1]
        st.text_area("出力", latest_result["result"], height=200, key="latest_output")
        
        csv_result = clean_csv(latest_result["result"])
        st.download_button(
            label="最新結果をCSVとしてダウンロード",
            data=csv_result,
            file_name="latest_result.csv",
            mime="text/csv",
            key="download_latest"
        )

if __name__ == "__main__":
    main()
