import json
import os
import time
import gradio as gr
from transformers import AutoModel, AutoTokenizer


class ChatModel():
    def __init__(self, model_path="model/chatglm2-6b"
                 , precision="int4",
                 use_gpu=True):

        path_root = os.path.dirname(__file__)
        model_path = os.path.join(path_root, model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision="")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, revision="")

        if not use_gpu:
            model = model.float()
        else:
            if precision == "fp16":
                model = model.half().cuda()
            elif precision == "int4":
                model = model.half().quantize(4).cuda()
            elif precision == "int8":
                model = model.half().quantize(8).cuda()

        self.model = model.eval()
        self.history = []

    def parse_codeblock(self, text):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "```" in line:
                if line != "```":
                    lines[i] = f'<pre><code class="{lines[i][3:]}">'
                else:
                    lines[i] = '</code></pre>'
            else:
                if i > 0:
                    lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
        return "".join(lines)

    def predict(self, query, max_length=2048, top_p=0.7, temperature=0.95):
        output, self.history = self.model.chat(
            self.tokenizer, query=query, history=self.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
        )
        output = self.parse_codeblock(output)
        return output, self.history

    def predictor(self, query, max_length=2048, top_p=0.7, temperature=0.95):

        pred=self.model.stream_chat(
            self.tokenizer, query=query, history=self.history,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
        )

        for p in pred:
            output,self.history=p
            output = self.parse_codeblock(output)
            yield output, self.history


    def clear_history(self):
        self.history.clear()

    def save_history(self):
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        s = [{"q": i[0], "o": i[1]} for i in self.history]
        filename = f"save-{int(time.time())}.json"
        with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False))

    def load_history(self, file):
        try:
            with open(file, "r", encoding='utf-8') as f:
                j = json.load(f)
                _hist = [(i["q"], i["o"]) for i in j]
                _readable_hist = [(i["q"], self.parse_codeblock(i["o"])) for i in j]

            self.history = _hist.copy()
        except Exception as e:
            print(e)

        return self.history


if __name__ == '__main__':
    model = ChatModel()
    print(model.predict("你好"))
    print(model.predict("你能够做什么"))
    print(model.predict("你能够读懂代码吗"))
