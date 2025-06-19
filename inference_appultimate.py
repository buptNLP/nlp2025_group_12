import os
import json
import paddle
import gradio
#print(f"脚本当前使用的Gradio版本是: {gradio.__version__}")
import gradio as gr
from PIL import Image
#from paddlenlp.transformers import ErnieMModel, ErnieMTokenizer
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from paddle.vision import transforms as T
from paddle.vision.models import resnet50
from paddle import nn
import paddle.nn.functional as F
import tempfile
import io
import pandas as pd
import requests
from bs4 import BeautifulSoup
import hashlib
import time
from readability import Document
from lxml import html
import uuid
import random
import re
import shutil

# ==============================================================================
# 0. 爬虫脚本配置与函数 (不变)
# ==============================================================================

# --- 配置项 ---
TEMP_BASE_INFERENCE_DIR = "temp_inference_data"
PERMANENT_SAVE_DIR = "saved_runs"
os.makedirs(TEMP_BASE_INFERENCE_DIR, exist_ok=True)
os.makedirs(PERMANENT_SAVE_DIR, exist_ok=True)


# ... (此处省略所有未改变的爬虫辅助函数) ...
def extract_text_and_images_from_url(url):
    for attempt in range(3):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            doc = Document(response.text)
            summary_html = doc.summary()
            parsed_body = html.fromstring(summary_html)
            main_content = parsed_body.text_content().strip()
            return main_content.strip(), []
        except Exception as e:
            print(f"Error processing URL {url} (Attempt {attempt + 1}/3): {e}")
            time.sleep(1)
    return "", []


def sogou_search(query_text, num_results=5):
    print(f"--- Running Sogou Search for: '{query_text}' ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = f"https://www.sogou.com/web?query={query_text}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        search_items = soup.find_all('div', class_='results')
        if not search_items: search_items = soup.find_all('div', class_='rb')
        for container in search_items:
            for result_div in container.find_all('div', class_=re.compile(r'vrwrap|news-box')):
                if len(results) >= num_results: break
                title_tag = result_div.find('h3')
                link_tag = title_tag.find('a') if title_tag else None
                snippet_tag = result_div.find('p', class_='fz-info') or result_div.find('span',
                                                                                        class_='text-info') or result_div.find(
                    'div', class_='citeurl')
                title = link_tag.get_text(strip=True) if link_tag else ""
                raw_link = link_tag.get('href') if link_tag else ""
                snippet = ' '.join(snippet_tag.get_text(strip=True).split()).strip() if snippet_tag else ""
                if not raw_link or not title: continue
                final_url = resolve_sogou_redirect(raw_link, headers)
                if final_url: results.append({"source_url": final_url, "title": title, "snippet": snippet})
        print(f"Found {len(results)} Sogou search results URLs.")
        return results
    except Exception as e:
        print(f"Error during Sogou search: {e}")
        return []


def resolve_sogou_redirect(url, headers):
    try:
        if url.startswith("/link?"): url = "https://www.sogou.com" + url
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response.raise_for_status()
        return response.url
    except Exception as e:
        print(f"Error resolving Sogou redirect {url}: {e}")
        return None


def search_for_relevant_info(query_text, num_results=5):
    print(f"--- Searching for: '{query_text}' ---")
    results = []
    try:
        sogou_results = sogou_search(query_text, num_results=num_results)
        for item in sogou_results:
            source_url = item.get('source_url')
            if not source_url: continue
            content, _ = extract_text_and_images_from_url(source_url)
            results.append({"source_url": source_url, "title": item.get('title'), "snippet": item.get('snippet', ''),
                            "content": content})
            time.sleep(random.uniform(0.5, 1.5))
    except Exception as e:
        print(f"Error during search processing: {e}")
    print(f"Total combined results: {len(results)}")
    return results


def extract_and_save_evidence_json(query_text, search_results, run_dir):
    annotation = {"all_fully_matched_captions": [], "all_partially_matched_captions": []}
    for res in search_results:
        title = res.get("title", "")
        snippet = res.get("snippet", "")
        content = res.get("content", "")
        if title: annotation["all_fully_matched_captions"].append({"caption": title, "url": res.get("source_url")})
        if snippet: annotation["all_fully_matched_captions"].append({"caption": snippet, "url": res.get("source_url")})
        if content: annotation["all_partially_matched_captions"].append(
            {"caption": content[:500] + "...", "url": res.get("source_url")})
    if not annotation["all_fully_matched_captions"] and not annotation["all_partially_matched_captions"]:
        return None, None
    direct_ann_file = os.path.join(run_dir, "direct_annotation.json")
    inverse_ann_file = os.path.join(run_dir, "inverse_annotation.json")
    with open(direct_ann_file, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    with open(inverse_ann_file, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)
    return direct_ann_file, inverse_ann_file


def collect_evidence_for_inference(query_text, query_image_path):
    item_id = str(uuid.uuid4())
    run_dir = os.path.join(TEMP_BASE_INFERENCE_DIR, item_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(query_image_path, os.path.join(run_dir, "query.jpg"))
    search_results = search_for_relevant_info(query_text)
    direct_ann_path, inverse_ann_path = extract_and_save_evidence_json(query_text, search_results, run_dir)
    if not direct_ann_path: return None
    return run_dir


# ==============================================================================
# 1. 模型定义与加载 (不变)
# ==============================================================================
class NetWork(nn.Layer):
    def __init__(self, mode="image"):
        super(NetWork, self).__init__()
        self.mode = mode
        # self.ernie = ErnieMModel.from_pretrained('ernie-m-base')
        # self.tokenizer = ErnieMTokenizer.from_pretrained('ernie-m-base')
        self.ernie = ErnieModel.from_pretrained('ernie-3.0-base-zh')
        self.tokenizer = ErnieTokenizer.from_pretrained('ernie-3.0-base-zh')
        self.attention_text = nn.MultiHeadAttention(embed_dim=768, num_heads=16)
        self.attention_image = nn.MultiHeadAttention(embed_dim=2048, num_heads=16)
        self.classifier1 = nn.Linear(2 * (768 + 2048), 1024)
        self.classifier2 = nn.Linear(1024, 3)

    def forward(self, qCap, qImg_feature, caps, imgs_features):
        encode_dict_qcap = self.tokenizer(text=qCap, max_length=128, truncation=True, padding='max_length')
        input_ids_qcap = paddle.to_tensor(encode_dict_qcap['input_ids'])
        qcap_feature, _ = self.ernie(input_ids_qcap)
        caps_feature = []
        for caption_list in caps:
            encode_dict_cap = self.tokenizer(text=caption_list, max_length=128, truncation=True, padding='max_length')
            input_ids_caps = paddle.to_tensor(encode_dict_cap['input_ids'])
            cap_feature, _ = self.ernie(input_ids_caps)
            cap_feature = cap_feature.mean(axis=1)
            caps_feature.append(cap_feature)
        caps_feature = paddle.stack(caps_feature, axis=0)
        caps_feature = self.attention_text(qcap_feature, caps_feature, caps_feature)
        qImg_feature = qImg_feature.unsqueeze(1)
        imgs_features = self.attention_image(qImg_feature, imgs_features, imgs_features)
        feature = paddle.concat([
            qcap_feature[:, 0, :], caps_feature[:, 0, :],
            qImg_feature.squeeze(1), imgs_features.squeeze(1)
        ], axis=-1)
        logits = self.classifier1(feature)
        logits = self.classifier2(logits)
        return logits


class EncoderCNN(nn.Layer):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2D((1, 1))

    def forward(self, x):
        out = self.resnet(x)
        out = self.adaptive_pool(out)
        return out.reshape([x.shape[0], -1])


print("正在加载模型，请稍候...")
model = NetWork(mode="image")
model_path = os.path.join("best_model", "model_best3.pdparams")
model.set_dict(paddle.load(model_path))
model.eval()
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
resnet_extractor = EncoderCNN()
resnet_extractor.eval()
print("模型加载完毕！")

# ==============================================================================
# 2. 核心推理与处理函数 (不变)
# ==============================================================================
label_map = {0: "✅ 真实", 1: "❌ 虚假", 2: "❓ 不确定"}


def run_prediction(query_img_pil, query_text_str, evidence_imgs_pil, evidence_texts_list):
    try:
        q_img_tensor = transform(query_img_pil).unsqueeze(0)
        qImg_feature = resnet_extractor(q_img_tensor)
        if not evidence_texts_list: evidence_texts_list = [query_text_str]
        evidence_features_list = []
        if evidence_imgs_pil:
            for img in evidence_imgs_pil:
                img_tensor = transform(img).unsqueeze(0)
                feature = resnet_extractor(img_tensor)
                evidence_features_list.append(feature)
        else:
            evidence_features_list.append(qImg_feature)
        caps_for_model = [evidence_texts_list]
        imgs_features_tensor = paddle.concat(evidence_features_list, axis=0).unsqueeze(0)
        logits = model(qCap=[query_text_str], qImg_feature=qImg_feature, caps=caps_for_model,
                       imgs_features=imgs_features_tensor)
        pred = paddle.argmax(F.softmax(logits, axis=-1), axis=1).item()
        return label_map[pred]
    except Exception as e:
        return f"⚠️ 推理失败: {str(e)}"


# ==============================================================================
# 3. 模式A和B的处理逻辑 (重构)
# ==============================================================================

# --- 模式A (不变) ---
def run_mode_a(q_img, q_text, direct_json_file, inverse_json_file, evidence_img_files):
    if not all([q_img, q_text, direct_json_file, inverse_json_file]):
        return None, "", pd.DataFrame(columns=["证据来源", "证据图片", "证据文本"]), "⚠️ 请上传所有必需的文件。"
    try:
        evidence_texts_for_model, evidence_imgs_for_model, evidence_for_display = [], [], []
        with open(direct_json_file.name, 'r', encoding='utf-8') as f:
            direct_data = json.load(f)
        uploaded_img_map = {os.path.basename(f.name): f.name for f in evidence_img_files} if evidence_img_files else {}
        for ev_list in direct_data.values():
            if isinstance(ev_list, list):
                for ev in ev_list:
                    title = ev.get('page_title', ev.get('title', '无标题'));
                    snippet = ev.get('snippet', '')
                    evidence_texts_for_model.append(f"{title} {snippet}");
                    display_text = f"**{title}**\n\n{snippet}"
                    img_path = uploaded_img_map.get(os.path.basename(ev.get('image_path', '')))
                    if img_path: evidence_imgs_for_model.append(Image.open(img_path).convert("RGB"))
                    evidence_for_display.append(["文本检索", img_path, display_text])
        with open(inverse_json_file.name, 'r', encoding='utf-8') as f:
            inverse_data = json.load(f)
        for cat in ["all_fully_matched_captions", "all_partially_matched_captions"]:
            for ev in inverse_data.get(cat, []):
                title = ev.get('title', '无标题')
                evidence_texts_for_model.append(title);
                evidence_for_display.append(["图片反向检索", None, f"**{title}**"])
        prediction_result = run_prediction(q_img, q_text, evidence_imgs_for_model, evidence_texts_for_model)
        evidence_df = pd.DataFrame(evidence_for_display, columns=["证据来源", "证据图片", "证据文本"])
        return q_img, q_text, evidence_df, prediction_result
    except Exception as e:
        return q_img, q_text, pd.DataFrame(), f"⚠️ 处理失败: {str(e)}"


# --- 模式B - 实时检索 ---
def run_mode_b_live(q_img, q_text):
    if not all([q_img, q_text]):
        return "请上传查询图片并输入查询文本。", " ", None, gr.update(visible=False), ""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        q_img.save(tmp.name)
        temp_query_image_path = tmp.name
    prediction_result, display_md, run_dir_state = " ", "正在搜集网络证据，请稍候...", None
    try:
        run_dir = collect_evidence_for_inference(q_text, temp_query_image_path)
        if not run_dir:
            return "❌ 证据搜集失败，可能网络超时或未找到相关结果。", " ", None, gr.update(visible=False), ""
        run_dir_state = run_dir
        evidence_texts_for_model, evidence_texts_for_display = [], []
        direct_ann_path = os.path.join(run_dir, "direct_annotation.json")
        if os.path.exists(direct_ann_path):
            with open(direct_ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cat in ["all_fully_matched_captions", "all_partially_matched_captions"]:
                for evidence in data.get(cat, []):
                    text, url = evidence.get('caption', ''), evidence.get('url', '#')
                    if text and text not in evidence_texts_for_model:
                        evidence_texts_for_model.append(text)
                        evidence_texts_for_display.append(f"- **[文本检索]** [{text}]({url})")
        prediction_result = run_prediction(q_img, q_text, None, evidence_texts_for_model)
        display_md = "\n".join(evidence_texts_for_display) if evidence_texts_for_display else "✅ 未找到明确的文本证据。"
    except Exception as e:
        display_md = f"⚠️ 实时验证过程中发生错误: {str(e)}"
    finally:
        if os.path.exists(temp_query_image_path): os.remove(temp_query_image_path)
    return display_md, prediction_result, run_dir_state, gr.update(visible=True if run_dir_state else False), ""


# --- 【新增】模式B - 手动输入 ---
def run_mode_b_manual(q_img, q_text, manual_evidence):
    if not all([q_img, q_text, manual_evidence]):
        return "请上传查询图片、输入查询文本和手动证据。", ""
    evidence_texts_list = [line.strip() for line in manual_evidence.strip().split('\n') if line.strip()]
    if not evidence_texts_list:
        return "手动证据内容不能为空。", ""
    prediction_result = run_prediction(q_img, q_text, None, evidence_texts_list)
    return f"已使用 **{len(evidence_texts_list)}** 条手动证据进行分析。", prediction_result


# --- 保存函数 (不变) ---
def save_run_results(run_dir, save_name):
    if not run_dir: return "❌ 无有效结果可保存。"
    if not save_name or not re.match(r'^[a-zA-Z0-9_-]+$', save_name):
        return "❌ 保存失败：请输入一个有效的保存名称（只允许字母、数字、下划线、连字符）。"
    try:
        target_dir = os.path.join(PERMANENT_SAVE_DIR, save_name)
        if os.path.exists(target_dir): return f"❌ 保存失败：名为 '{save_name}' 的目录已存在。"
        shutil.copytree(run_dir, target_dir)
        return f"✅ 结果已成功保存到目录: {target_dir}"
    except Exception as e:
        return f"❌ 保存失败: {str(e)}"

# ==============================================================================
# 4. Gradio 双模式界面构建 (最终修正版)
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 双模式多模态谣言检测系统")
    gr.Markdown("在下方选项卡中选择不同的验证模式。")

    with gr.Tabs():
        # --- 模式 A ---
        with gr.TabItem("模式A：多模态验证 (使用本地上传文件)"):
            gr.Markdown("此模式用于演示模型在**双路证据（文本检索+图片反搜）齐全**下的综合判断能力。")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1. 上传查询内容");
                    mode_a_q_img = gr.Image(type="pil", label="上传查询图片");
                    mode_a_q_text = gr.Textbox(lines=3, placeholder="请输入查询文本...", label="查询文本")
                with gr.Column(scale=3):
                    gr.Markdown("### 2. 上传所有证据文件");
                    mode_a_direct_json = gr.File(label="上传 direct_annotation.json (文本检索证据)",
                                                 file_types=[".json"]);
                    mode_a_inverse_json = gr.File(label="上传 inverse_annotation.json (图片反搜证据)",
                                                  file_types=[".json"]);
                    mode_a_evidence_imgs = gr.File(label="上传 direct_annotation.json 中对应的所有证据图片",
                                                   file_count="multiple", file_types=["image"])
            mode_a_button = gr.Button("开始验证", variant="primary")
            gr.Markdown("---");
            gr.Markdown("### 3. 证据详情")
            mode_a_evidence_df = gr.DataFrame(headers=["证据来源", "证据图片", "证据文本"],
                                              datatype=["str", "image", "markdown"], label="所有证据详情")
            gr.Markdown("---");
            gr.Markdown("### 4. 模型预测最终结果")
            mode_a_prediction = gr.Label(label="最终结果", show_label=False)

        # --- 模式 B ---
        with gr.TabItem("模式B：实时/手动文本验证"):
            mode_b_run_dir_state = gr.State(None)
            gr.Markdown("此模式用于演示系统对**纯文本证据**的处理能力，支持**实时在线检索**或**手动输入**两种方式。")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 上传/输入查询内容");
                    mode_b_q_img = gr.Image(type="pil", label="上传查询图片");
                    mode_b_q_text = gr.Textbox(lines=5, placeholder="请输入待查询的文本...", label="查询文本")
                with gr.Column(scale=2):
                    gr.Markdown("### 2. 选择证据来源方式")
                    mode_b_choice = gr.Radio(choices=["实时在线检索", "手动输入文本证据"], label="证据来源",
                                             value="实时在线检索")

                    # 【最终修正】使用正确的 gr.Group 组件
                    with gr.Group(visible=True) as mode_b_live_box:
                        mode_b_live_button = gr.Button("开始实时核查", variant="primary")
                        gr.Markdown("#### 实时检索到的文本证据");
                        mode_b_live_display = gr.Markdown(label="检索结果将显示于此")
                        with gr.Row(visible=False) as mode_b_save_row:
                            with gr.Column(scale=2): mode_b_save_name = gr.Textbox(label="输入保存名称",
                                                                                   placeholder="例如：case_001")
                            with gr.Column(scale=1): mode_b_save_button = gr.Button("💾 保存本次结果")
                        mode_b_save_status = gr.Markdown()

                    # 【最终修正】使用正确的 gr.Group 组件
                    with gr.Group(visible=False) as mode_b_manual_box:
                        mode_b_manual_input = gr.Textbox(lines=8, placeholder="请输入所有证据文本，每条证据占一行...",
                                                         label="手动输入证据文本")
                        mode_b_manual_button = gr.Button("使用手动证据验证", variant="primary")
                        mode_b_manual_display = gr.Markdown(label="手动验证说明")

            gr.Markdown("---")
            gr.Markdown("### 3. 模型预测结果")
            mode_b_prediction = gr.Label(label="模型预测结果")

    # --- 绑定按钮与函数 (不变) ---
    mode_a_button.click(fn=run_mode_a, inputs=[mode_a_q_img, mode_a_q_text, mode_a_direct_json, mode_a_inverse_json,
                                               mode_a_evidence_imgs],
                        outputs=[mode_a_q_img, mode_a_q_text, mode_a_evidence_df, mode_a_prediction])
    mode_b_live_button.click(fn=run_mode_b_live, inputs=[mode_b_q_img, mode_b_q_text],
                             outputs=[mode_b_live_display, mode_b_prediction, mode_b_run_dir_state, mode_b_save_row,
                                      mode_b_save_status])
    mode_b_save_button.click(fn=save_run_results, inputs=[mode_b_run_dir_state, mode_b_save_name],
                             outputs=[mode_b_save_status])
    mode_b_manual_button.click(fn=run_mode_b_manual, inputs=[mode_b_q_img, mode_b_q_text, mode_b_manual_input],
                               outputs=[mode_b_manual_display, mode_b_prediction])


    def switch_mode_b_ui(choice):
        if choice == "实时在线检索":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)


    mode_b_choice.change(fn=switch_mode_b_ui, inputs=mode_b_choice, outputs=[mode_b_live_box, mode_b_manual_box])

if __name__ == "__main__":
    print("\nGradio 界面启动中，请在浏览器中打开给出的链接。")
    demo.launch(share=True)