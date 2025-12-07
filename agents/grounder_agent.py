# -*- coding: utf-8 -*-
# 文件路径: SymbolGeneration/Agent/agents/grounder_agent.py
from __future__ import annotations
import json, re, requests
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI

from ..utils import log, save_json, extract_json
from ..config import OPENAI_API_KEY, MODELS

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Endpoints ---
WIKI_SEARCH = "https://{lang}.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


# [关键函数] 百度图片搜索 (JSON API 版)
def _search_baidu_image(keyword: str) -> Optional[str]:
    """
    使用百度图片搜索的后台 JSON 接口 (acjson)。
    无需翻墙，解析稳定，直接返回图片 URL。
    """
    print(f"🔎 [Baidu] 正在搜索图片: {keyword}")
    try:
        url = "https://image.baidu.com/search/acjson"

        # 伪装成浏览器的滚动加载请求
        params = {
            "tn": "resultjson_com",
            "logid": "8305096434442765369",
            "ipn": "rj",
            "ct": "201326592",
            "is": "",
            "fp": "result",
            "queryWord": keyword,
            "cl": "2",
            "lm": "-1",
            "ie": "utf-8",
            "oe": "utf-8",
            "adpicid": "",
            "st": "-1",
            "z": "",
            "ic": "0",
            "hd": "",
            "latest": "",
            "copyright": "",
            "word": keyword,
            "s": "",
            "se": "",
            "tab": "",
            "width": "",
            "height": "",
            "face": "0",
            "istype": "2",
            "qc": "",
            "nc": "1",
            "fr": "",
            "expermode": "",
            "force": "",
            "pn": "0",
            "rn": "30",
            "gsm": "1e",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/plain, */*; q=0.01",
            "Referer": "https://image.baidu.com/search/index",
            "X-Requested-With": "XMLHttpRequest",
        }

        res = requests.get(url, params=params, headers=headers, timeout=8)

        if res.status_code == 200:
            try:
                # 处理非标准 JSON 的转义字符
                json_str = res.text.replace(r"\'", "'")
                data = json.loads(json_str)

                if "data" not in data or not isinstance(data["data"], list):
                    return None

                candidates = []

                # 1. 收集候选图 (遍历所有返回的 30 张图)
                for item in data["data"]:
                    if not isinstance(item, dict): continue

                    # 优先取 thumbURL (缩略图，链接稳定)
                    img_url = item.get("thumbURL") or item.get("middleURL")
                    if not img_url: continue

                    # 获取尺寸信息
                    w = int(item.get("width", 0) or 0)
                    h = int(item.get("height", 0) or 0)

                    if w > 200 and h > 200:
                        print(f"✅ [Baidu] 选中首张清晰图片: {img_url[:50]}...")
                        return img_url

                # 2. [智能筛选] 优先找横构图 (长宽比 > 1.2)
                # 这种图片通常是地标的全景照，能让 Detector 识别出"躺着"
                best_match = None
                for cand in candidates:
                    # 过滤太小的图
                    if cand["w"] < 200 or cand["h"] < 150: continue

                    # 关键条件：必须是横向的
                    if cand["ratio"] > 1.2:
                        best_match = cand["url"]
                        print(f"✅ [Smart Pick] 选中横向全景图 (W:{cand['w']} H:{cand['h']}): {best_match[:50]}...")
                        break

                # 3. 兜底：如果全是竖图，没办法，只能用第一张
                if not best_match and candidates:
                    best_match = candidates[0]["url"]
                    print(f"⚠️ [Fallback] 未找到完美构图，使用首张结果: {best_match[:50]}...")

                return best_match

            except Exception as e:
                print(f"⚠️ 百度返回数据解析失败: {e}")
                pass

    except Exception as e:
        print(f"⚠️ 百度搜图失败: {e}")

    return None


def _gather_raw_knowledge(user_text: str, search_focus: str = None) -> Tuple[str, Optional[str]]:
    queries = _expand_queries(user_text)
    blobs = []
    first_image = None

    # 如果有精准搜索词，把它加到查询列表的最前面！
    if search_focus:
        queries.insert(0, search_focus)

    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text)

    for q in queries:
        # 1. 尝试百度百科
        if has_chinese:
            summary, img = _fetch_baidu_baike(q)
            if summary:
                blobs.append(f"[Baidu] {q}\n{summary}")
                if not first_image and img: first_image = img

                # 如果百科没图，用当前的 query (q) 去搜图
                if not first_image:
                    first_image = _search_baidu_image(q)
                continue

                # 2. 维基百科逻辑 (保持不变)
        # ... (略，保持原代码) ...

    # 3. [核心修改] 最终兜底：优先使用精准词搜图，而不是用长句子
    if not first_image and has_chinese:
        # 如果有 search_focus (如"兰州白塔山")，用它搜！
        target_keyword = search_focus if search_focus else user_text
        print(f"🔎 最终兜底：尝试使用关键词搜索图片: {target_keyword}")
        first_image = _search_baidu_image(target_keyword)

    text = "\n\n".join(blobs)
    log("Grounder_raw", text if text else "(empty)")

    return text, first_image


# [修改] 接口增加 search_focus
def ground_entity_to_spec(user_text: str, search_focus: str = None) -> Dict[str, Any]:
    # 传递 search_focus 给搜图逻辑
    raw_text, ref_image_url = _gather_raw_knowledge(user_text, search_focus=search_focus)

    if not raw_text and not ref_image_url:
        spec = {"entity": {"name": user_text}, "constraints": {"must_not": []}}
        save_json("Grounder_spec", spec)
        return spec

    # ... (中间 LLM 调用代码保持不变) ...
    msg_user = [
        {"type": "text", "text": f"User intent:\n{user_text}"},
        {"type": "text", "text": f"Raw encyclopedia snippets:\n{raw_text}"}
    ]
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_TO_SPEC}, {"role": "user", "content": msg_user}]
    )
    spec = extract_json(resp.choices[0].message.content) or {"entity": {"name": user_text}}

    if not spec.get("constraints"): spec["constraints"] = {}
    spec["constraints"].setdefault("must_not", [])

    if ref_image_url:
        spec["reference_image_url"] = ref_image_url

    save_json("Grounder_spec", spec)
    return spec


# ----------------- Baidu Baike Helper -----------------
def _fetch_baidu_baike(keyword: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://baike.baidu.com/item/{keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        if resp.status_code != 200:
            return None, None

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 1. 提取文本
        texts = []
        summary_div = soup.find('div', class_='lemma-summary')
        if summary_div:
            texts.append(summary_div.get_text().strip())

        basic_info = soup.find('div', class_='basic-info')
        if basic_info:
            names = basic_info.find_all('dt')
            values = basic_info.find_all('dd')
            for n, v in zip(names, values):
                texts.append(f"{n.get_text().strip()}: {v.get_text().strip()}")

        summary_text = "\n".join(texts)
        if not summary_text: return None, None

        # 2. 尝试从百科提取图片 (仅作为尝试)
        image_url = None
        meta_img = soup.find('meta', property="og:image")
        if meta_img:
            image_url = meta_img.get("content")

        if not image_url:
            pic_div = soup.find('div', class_='summary-pic')
            if pic_div:
                img = pic_div.find('img')
                if img: image_url = img.get('src')

        if image_url:
            if image_url.startswith('//'):
                image_url = "https:" + image_url
            elif image_url.startswith('/'):
                image_url = "https://baike.baidu.com" + image_url

        return summary_text, image_url

    except Exception as e:
        print(f"⚠️ Baidu Baike fetch error: {e}")
        return None, None


# ----------------- Small Helpers (Wiki) -----------------
def _wiki_search(q: str, lang="en") -> Optional[str]:
    try:
        params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}
        r = requests.get(WIKI_SEARCH.format(lang=lang), params=params, timeout=5)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, list) and len(j) >= 2 and j[1]: return j[1][0]
    except Exception:
        pass
    return None


def _wiki_summary(title: str, lang="en") -> Optional[Dict[str, Any]]:
    try:
        url = WIKI_SUMMARY.format(lang=lang, title=title.replace(" ", "_"))
        r = requests.get(url, timeout=5, headers={"accept": "application/json"})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _expand_queries(user_text: str) -> List[str]:
    qs: List[str] = [user_text.strip()]
    for seg in re.findall(r"[一-龥A-Za-z0-9·\-\s]{2,}", user_text):
        s = seg.strip()
        if s and s not in qs: qs.append(s)
    return list(dict.fromkeys(qs))


def _langs_for(q: str, user_text: str) -> List[str]:
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text + q)
    return ["zh", "en"] if has_chinese else ["en", "zh"]


# ----------------- Main Logic -----------------
def _gather_raw_knowledge(user_text: str) -> Tuple[str, Optional[str]]:
    queries = _expand_queries(user_text)
    blobs = []
    first_image = None

    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text)

    for q in queries:
        # 1. 尝试百度百科
        if has_chinese:
            summary, img = _fetch_baidu_baike(q)
            if summary:
                blobs.append(f"[Baidu] {q}\n{summary}")

                # 如果百科有图，暂存
                if not first_image and img:
                    first_image = img

                # [关键] 如果百科有文但没图，调用百度图片搜索补救
                if not first_image:
                    first_image = _search_baidu_image(q)

                continue

                # 2. 尝试维基百科
        langs = _langs_for(q, user_text)
        for lang in langs:
            title = _wiki_search(q, lang)
            if title:
                data = _wiki_summary(title, lang)
                if data:
                    extract = data.get("extract")
                    img_src = data.get("thumbnail", {}).get("source") or data.get("originalimage", {}).get("source")

                    if extract:
                        blobs.append(f"[Wiki-{lang}] {title}\n{extract}")
                        if not first_image and img_src:
                            first_image = img_src
                        break

    # 3. [最后兜底] 仍然没图？用原词去百度图片搜一把
    if not first_image and has_chinese:
        print(f"🔎 最终兜底：尝试使用百度搜索图片: {user_text}")
        first_image = _search_baidu_image(user_text)

    text = "\n\n".join(blobs)
    log("Grounder_raw", text if text else "(empty)")

    if first_image:
        log("Grounder_image", first_image)

    return text, first_image


# (SYSTEM_TO_SPEC 保持不变)
SYSTEM_TO_SPEC = (
    "You are a visual knowledge extraction expert. "
    "Your task is to convert vague user intent and raw encyclopedia snippets into a STRICT visual structure spec.\n"
    "Goal: Extract specific physical constraints so a blind painter can reconstruct the landmark accurately.\n"
    "Schema:\n"
    "{ \n"
    "  \"entity\": {\"name\": str, \"location\": str},\n"
    "  \"entity_type\": \"bridge|tower|building|statue|logogram|other\",\n"
    "  \"structure\": {\n"
    "      \"structural_system\": \"truss|arch|suspension|beam|unknown\",\n"
    "      \"shape_features\": [str],  // e.g. \"3 spans\", \"octagonal base\", \"reclining posture\"\n"
    "      \"material\": \"steel|stone|concrete|wood\",\n"
    "      \"view_recommendation\": \"side|front|isometric\"\n"
    "  },\n"
    "  \"constraints\": {\n"
    "      \"must\": [str],      // Visual elements that MUST appear\n"
    "      \"must_not\": [str]   // Elements to EXCLUDE\n"
    "  }\n"
    "}\n"
    "Rules:\n"
    "1. Rely HEAVILY on the provided snippets.\n"
    "2. If snippets describe a statue, extract posture and composition details.\n"
    "3. Return ONLY a JSON object."
)


def ground_entity_to_spec(user_text: str) -> Dict[str, Any]:
    raw_text, ref_image_url = _gather_raw_knowledge(user_text)

    if not raw_text and not ref_image_url:
        spec = {"entity": {"name": user_text}, "constraints": {"must_not": []}}
        save_json("Grounder_spec", spec)
        return spec

    msg_user = [
        {"type": "text", "text": f"User intent:\n{user_text}"},
        {"type": "text", "text": f"Raw encyclopedia snippets:\n{raw_text}"}
    ]
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_TO_SPEC}, {"role": "user", "content": msg_user}]
    )
    spec = extract_json(resp.choices[0].message.content) or {"entity": {"name": user_text}}

    if not spec.get("constraints"): spec["constraints"] = {}
    spec["constraints"].setdefault("must_not", [])

    if ref_image_url:
        spec["reference_image_url"] = ref_image_url

    save_json("Grounder_spec", spec)
    return spec