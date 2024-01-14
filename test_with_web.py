import gradio as gr
from chat_app import ChatApp

_css = """
#del-btn {
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.0em;
    margin: 1.5em 0;
}
"""

app = ChatApp(sentence_embedding_model_path="model/EmbeddingModel/all-MiniLM-L6-v2")  # 加载模型
t = None

def save_history():
    app.save_history()


def load_history(file):
    history = app.load_history(file)
    return history


def clear_history():
    app.clear_history()
    return gr.update(value=[])

def predict(text, history):
    return "", history + [[text, ""]]#点击发送，清空输入，chatbot更新显示

def bot(history,max_length=2048, top_p=0.7, temperature=0.95):
    q=history[-1][0]
    predictor= app.stream_query(q,max_length, top_p, temperature)
    for output in predictor:
        yield output

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        prompt = "请输入你的内容(Shift + Enter = 发送, Enter = 换行)"
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""<h2><center>诸城党建小助手</center></h2>""")
                with gr.Accordion("模型参数调整（保持默认值即可）",open=False):
                    with gr.Row():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                max_length = gr.Slider(minimum=4, maximum=4096, step=4, label='Max Length', value=2048)
                                top_p = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Top P', value=0.7)
                            with gr.Row():
                                temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Temperature',
                                                        value=0.95)

                with gr.Row():
                    with gr.Column(variant="panel"):
                        with gr.Row():
                            clear = gr.Button("清空对话（上下文）")

                        with gr.Row():
                            save_his_btn = gr.Button("保存对话")
                            load_his_btn = gr.UploadButton("读取对话", file_types=['file'], file_count='single')

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=600)
                with gr.Row():
                    message = gr.Textbox(placeholder=prompt, show_label=False, lines=2, scale=20)
                    with gr.Column(scale=1):
                        submit = gr.Button("发送")
                        message.submit(predict, inputs=[message, chatbot], outputs=[message, chatbot],
                                       queue=False).then(bot, [chatbot,max_length,top_p,temperature], chatbot)
                        clear_input = gr.Button("删除", elem_id="del-btn")

        submit.click(predict, inputs=[message, chatbot], outputs=[message, chatbot],
                     queue=False).then(bot, [chatbot,max_length,top_p,temperature], chatbot)

        clear.click(clear_history, outputs=[chatbot])
        clear_input.click(lambda x: "", inputs=[message], outputs=[message])

        save_his_btn.click(save_history)
        load_his_btn.upload(load_history, inputs=[
            load_his_btn,
        ], outputs=[
            chatbot
        ])

    return demo


ui = create_ui()
ui.queue().launch(
    server_name="127.0.0.1",
    server_port=8080,
    share=True
)
