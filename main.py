import gradio as gr
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ------------------------------------------------------------------
#  環境変数の読み込み
# ------------------------------------------------------------------
# ローカル開発用に.envファイルから環境変数を読み込む
# Render環境ではこの行は無視され、Renderの環境変数が使われる
load_dotenv()

# 環境変数からAPIキー、ユーザー名、パスワードを取得
api_key = os.environ.get('GOOGLE_API_KEY')
gradio_user = os.environ.get('GRADIO_USERNAME')
gradio_pass = os.environ.get('GRADIO_PASSWORD')

# APIキーが設定されていない場合はエラーで終了
if not api_key:
    print("エラー: 環境変数にGOOGLE_API_KEYを設定してください。")
    exit()

genai.configure(api_key=api_key)

# ------------------------------------------------------------------
#  モデルの設定
# ------------------------------------------------------------------
model = genai.GenerativeModel('gemini-2.5-flash')

# ------------------------------------------------------------------
#  応答を生成する関数
# ------------------------------------------------------------------
def generate_response(message, history):
    gemini_history = []
    for user_msg, model_msg in history:
        gemini_history.append({'role': 'user', 'parts': [user_msg]})
        gemini_history.append({'role': 'model', 'parts': [model_msg]})
    
    chat = model.start_chat(history=gemini_history)
    
    response = chat.send_message(message, stream=True)
    
    full_response = ""
    for chunk in response:
        full_response += chunk.text
        yield full_response

# ------------------------------------------------------------------
#  Gradio UIの作成と起動
# ------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=generate_response,
    title="🤖 Geminiチャット (Render版)",
    description="Gemini 2.5 Flashモデルとチャットできます。",
)

# アプリを起動
if __name__ == "__main__":
    # Renderが指定するポート番号を取得。なければ7860をデフォルトに
    port = int(os.environ.get('PORT', 7860))
    
    # 認証情報が環境変数に設定されている場合のみ、認証を有効にする
    auth_credentials = (gradio_user, gradio_pass) if gradio_user and gradio_pass else None

    # Renderで公開するためにserver_name="0.0.0.0"を指定
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        auth=auth_credentials
    )