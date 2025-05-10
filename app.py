"""Emoberta app with Ngrok tunnel integration"""
import argparse
import logging
import os
import jsonpickle
import torch
import threading
from flask import Flask, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyngrok import ngrok

# 配置基础日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------- GLOBAL VARIABLES ---------------------- #
emotions = [
    "neutral",
    "joy",
    "surprise",
    "anger",
    "sadness",
    "disgust",
    "fear",
]
id2emotion = {idx: emotion for idx, emotion in enumerate(emotions)}
tokenizer = None
model = None
device = None
app = Flask(__name__)
# --------------------------------------------------------------- #

def load_tokenizer_model(model_type: str, device_: str) -> None:
    """Load tokenizer and model.
    Args
    ----
    model_type: Should be either "emoberta-base" or "emoberta-large"
    device_: "cpu" or "cuda"
    """
    if "large" in model_type.lower():
        model_type = "emoberta-large"
    elif "base" in model_type.lower():
        model_type = "emoberta-base"
    else:
        raise ValueError(
            f"{model_type} is not a valid model type! Should be 'base' or 'large'."
        )

    if not os.path.isdir(model_type):
        model_type = f"tae898/{model_type}"

    global device
    device = device_

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    global model
    model = AutoModelForSequenceClassification.from_pretrained(model_type)
    model.eval()
    model.to(device)
    logging.info(f"Model {model_type} loaded successfully on {device}")

@app.route("/", methods=["POST"])
def run_emoberta():
    """Receive everything in json!!!"""
    app.logger.debug("Receiving data ...")
    data = request.json
    data = jsonpickle.decode(data)
    text = data["text"]
    app.logger.info(f"raw text received: {text}")

    tokens = tokenizer(text, truncation=True)
    tokens["input_ids"] = torch.tensor(tokens["input_ids"]).view(1, -1).to(device)
    tokens["attention_mask"] = (
        torch.tensor(tokens["attention_mask"]).view(1, -1).to(device)
    )

    outputs = model(**tokens)
    outputs = torch.softmax(outputs["logits"].detach().cpu(), dim=1).squeeze().numpy()
    outputs = {id2emotion[idx]: prob.item() for idx, prob in enumerate(outputs)}

    app.logger.info(f"prediction: {outputs}")
    response = jsonpickle.encode(outputs)
    app.logger.info("json-pickle is done.")

    return response

@app.route("/", methods=["GET"])
def home():
    """添加一个简单的GET方法路由，用于测试服务是否运行"""
    return """
    <html>
        <head><title>Emoberta API</title></head>
        <body>
            <h1>Emoberta API 服务正在运行</h1>
            <p>这是一个情感分析API。请使用POST请求并提供JSON格式的文本数据。</p>
            <p>示例：{"text": "我感到非常开心！"}</p>
        </body>
    </html>
    """

def setup_ngrok(port, auth_token=None):
    """设置并启动Ngrok隧道"""
    # 如果提供了认证令牌，则设置它
    if auth_token:
        ngrok.set_auth_token(auth_token)

    # 创建一个HTTP隧道
    public_url = ngrok.connect(port)
    logging.info(f"Ngrok隧道已创建: {public_url}")

    return public_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="emoberta app with Ngrok tunnel.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host ip address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10006,
        help="port number",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="emoberta-base",
        help="should be either emoberta-base or emoberta-large",
    )
    parser.add_argument(
        "--use-ngrok",
        action="store_true",
        help="enable ngrok tunnel",
    )
    parser.add_argument(
        "--ngrok-token",
        type=str,
        default=None,
        help="ngrok authentication token (optional)",
    )

    args = parser.parse_args()

    # 加载模型
    load_tokenizer_model(args.model_type, args.device)

    # 如果启用了Ngrok
    if args.use_ngrok:
        # 确保pyngrok已安装
        try:
            import pyngrok
        except ImportError:
            logging.error("pyngrok未安装。请使用'pip install pyngrok'安装它。")
            exit(1)

        # 设置Ngrok隧道
        public_url = setup_ngrok(args.port, args.ngrok_token)
        logging.info(f"请通过以下地址访问Emoberta API: {public_url}")

    # 启动Flask应用
    app.run(host=args.host, port=args.port)