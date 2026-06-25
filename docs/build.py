#!/usr/bin/env python3
"""Scan glassbox cv/ and llm/ component folders and bake a self-contained
mastery-map page (docs/index.html). Stdlib only, no third-party deps.

Run:  .venv/bin/python docs/build.py   (then open docs/index.html)
"""
import json
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent      # glassbox/
OUT = pathlib.Path(__file__).resolve().parent / "index.html"

# component dir -> (VLM-anatomy zone, Qwen-VL module, replaces, who-uses-it)
# Hand-curated semantic placement; README / code / mastery are read from the repo.
MANIFEST = {
    "cv/vit":         ("vision",   "ViT · patch embed",       "CNN（卷积归纳偏置）",      "所有 VLM 视觉塔"),
    "cv/clip":        ("vision",   "CLIP 式对齐训练",          "有监督标签预训练",        "CLIP · SigLIP · 多数 VLM"),
    "cv/siglip":      ("vision",   "sigmoid 对齐目标",         "CLIP 的 softmax 对比 loss", "SigLIP · 2024+ VLM"),
    "cv/mae":         ("vision",   "掩码自监督预训练",         "有标签预训练",            "MAE 系视觉自监督"),
    "cv/videomae":    ("vision",   "视频掩码自监督",           "图像级自监督",            "VideoMAE"),
    "cv/vl":          ("join",     "MLP projector · splice",   "原样塞所有 patch token",   "LLaVA / Qwen-VL"),
    "cv/mrope":       ("join",     "M-RoPE 多模态位置",        "1D RoPE 处理多模态",       "Qwen2-VL / 2.5-VL"),
    "cv/tokmerge":    ("budget",   "2×2 token 合并",           "传全部 patch token",       "Qwen-VL merger"),
    "cv/framesample": ("budget",   "dynamic FPS 采样",         "固定均匀抽帧",            "所有视频 VLM"),
    "llm/bpe":        ("language", "BPE tokenizer",            "字符 / 词级切分",          "GPT · Llama · Qwen"),
    "llm/rope":       ("language", "RoPE 位置编码",            "正弦绝对位置",            "几乎所有现代 LLM"),
    "llm/rmsnorm":    ("language", "RMSNorm",                  "LayerNorm",               "Llama · Qwen"),
    "llm/swiglu":     ("language", "SwiGLU FFN",               "ReLU / GELU FFN",          "Llama · Qwen"),
    "llm/gqa":        ("language", "GQA",                      "MHA（多头全 KV）",         "Llama · Qwen"),
    "llm/nano":       ("language", "组装 · 小型 causal LLM",   "—",                        "把上面拼成可训练 LLM"),
    "llm/kvcache":    ("inference","推理 KV 缓存",             "每步重算",                "所有自回归推理"),
    "llm/localattn":  ("inference","局部 / 滑窗注意力",        "全 O(n²) 注意力",          "长上下文 · Qwen ViT 也用"),
    "llm/moe":        ("language", "MoE 稀疏 FFN",             "dense FFN",               "Qwen-MoE · DeepSeek"),
    "llm/mla":        ("language", "MLA 潜在注意力",           "MHA / GQA 的 KV",          "DeepSeek-V2/V3"),
    "llm/specdec":    ("inference","投机解码",                 "逐 token 解码",            "推理加速"),
    "cv/tokbudget":   ("budget",   "全链 token 账本",          "凭感觉估 token",           "长视频系统设计"),
}

ZONES = [
    ("language",  "llm · 语言骨干",       "tokenizer、位置、norm、attention、FFN、nano 组装"),
    ("vision",    "cv · 视觉塔",           "pixels / frames → patch tokens；图文预训练目标"),
    ("join",      "VLM 接缝",              "projector + splice + multimodal positions"),
    ("budget",    "视觉 token 预算支线",    "抽帧、合并、成本账本：控制进 LLM 的视觉序列长度"),
    ("inference", "LLM 推理支线",          "cache、局部注意力、投机解码：服务与长上下文机制"),
    ("other",     "其他 / 未归类",          "manifest 未归类的组件"),
]

SKIP_STEMS = {"experiment", "prepare", "train_sanity", "build"}


def first_title(md, fallback):
    m = re.search(r"^#\s+(.+)$", md, re.M)
    return m.group(1).strip() if m else fallback


def one_liner(md):
    m = re.search(r"一句话[^\n]*\n+([^\n#>].+)", md)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:160]
    parts = re.split(r"^#\s+.+$", md, maxsplit=1, flags=re.M)
    if len(parts) > 1:
        for para in parts[1].split("\n\n"):
            p = para.strip()
            if p and p[0] not in ">#`|-*":
                return re.sub(r"\s+", " ", p)[:160]
    return ""


def scan(folder):
    d = ROOT / folder
    readme = d / "README.md"
    md = readme.read_text(encoding="utf-8") if readme.exists() else ""
    pys = list(d.glob("*.py"))
    impl = [p for p in pys if p.stem not in SKIP_STEMS]
    has_py = bool(impl)
    has_exp = any(p.name == "experiment.py" for p in pys) or "## Experiment" in md
    english = "## Experiment" in md or "> **References**" in md
    if not has_py:
        mastery = "spec"
    elif has_exp or english:
        mastery = "explained"
    else:
        mastery = "core"
    name = folder.split("/")[-1]
    src = next((p for p in impl if p.stem == name), impl[0] if impl else None)
    return {
        "key": folder,
        "name": name,
        "title": first_title(md, name),
        "oneliner": one_liner(md),
        "mastery": mastery,
        "readme": md,
        "code": src.read_text(encoding="utf-8") if src else "",
        "code_file": src.name if src else None,
    }


def collect():
    comps, known = [], set(MANIFEST)
    for key, (zone, qwen, rep, who) in MANIFEST.items():
        if not (ROOT / key).exists():
            continue
        c = scan(key)
        c.update(zone=zone, qwen=qwen, replaces=rep, who=who)
        comps.append(c)
    for line in ("cv", "llm"):
        for d in sorted((ROOT / line).glob("*/")):
            key = f"{line}/{d.name}"
            if key not in known and (d / "README.md").exists():
                c = scan(key)
                c.update(zone="other", qwen="", replaces="", who="")
                comps.append(c)
    return comps


def main():
    comps = collect()
    data = {"zones": ZONES, "components": comps}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")
    counts = {"explained": 0, "core": 0, "spec": 0}
    for c in comps:
        counts[c["mastery"]] = counts.get(c["mastery"], 0) + 1
    print(f"scanned {len(comps)} components -> {OUT}")
    print(f"  implemented+explained {counts['explained']} · implemented-core {counts['core']} · spec-only {counts['spec']}")
    for c in comps:
        print(f"  [{c['mastery']:9}] {c['key']:16} {c['code_file'] or '(no code)'}")


TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>glassbox VLM component map</title>
<link rel="stylesheet" media="(prefers-color-scheme: light)" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<link rel="stylesheet" media="(prefers-color-scheme: dark)" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<style>
:root{
  --bg:#faf9f5; --surface:#ffffff; --line:rgba(0,0,0,.12); --line2:rgba(0,0,0,.22);
  --text:#1a1a18; --muted:#6b6a64; --hint:#94938c;
  --done:#1d9e75; --core:#ba7517; --spec:#b4b2a9;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#1f1e1c; --surface:#2a2926; --line:rgba(255,255,255,.14); --line2:rgba(255,255,255,.26);
    --text:#ecebe6; --muted:#a6a59e; --hint:#7a7972;
    --done:#5dcaa5; --core:#ef9f27; --spec:#5f5e5a;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Hiragino Sans GB",sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:2rem 1.25rem 4rem;display:grid;grid-template-columns:1fr;gap:1.25rem}
@media(min-width:880px){.wrap{grid-template-columns:1.35fr 1fr}.head,.zones{grid-column:1}.detail{grid-column:2;grid-row:1/99}}
h1{font-size:22px;font-weight:600;margin:0}
.sub{font-size:13px;color:var(--muted);margin-top:3px}
.seg{height:9px;border-radius:99px;display:flex;overflow:hidden;background:var(--surface);border:.5px solid var(--line);margin-top:14px}
.seg>span{display:block}
.cnt{font-size:12px;color:var(--muted);margin-top:7px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:9px;font-size:12px;color:var(--muted);align-items:center}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none;margin-right:5px;vertical-align:middle}
.zone{margin-top:1.4rem}
.zh{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.zh b{font-weight:600;font-size:15px}
.zh span{font-size:12px;color:var(--hint)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.card{background:var(--surface);border:.5px solid var(--line);border-radius:12px;padding:11px 13px;cursor:pointer;transition:border-color .15s}
.card:hover{border-color:var(--line2)}
.card.sel{border-color:var(--done);border-width:1px}
.card .top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.card .nm{font-size:14px;font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.card .ol{font-size:12px;color:var(--muted);margin-top:5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.detail{background:var(--surface);border:.5px solid var(--line);border-radius:14px;padding:1.1rem 1.25rem;align-self:start;position:sticky;top:1rem;max-height:calc(100vh - 2rem);overflow:auto}
.detail h2{font-size:17px;margin:0 0 2px;font-family:ui-monospace,Menlo,monospace}
.detail .meta{font-size:12.5px;color:var(--muted);margin:8px 0 4px}
.detail .row{display:flex;gap:10px;font-size:13px;padding:3px 0;border-top:.5px solid var(--line)}
.detail .row span:first-child{color:var(--hint);min-width:52px;flex:none}
.tabs{display:flex;gap:6px;margin:14px 0 10px}
.tab{font-size:12px;padding:4px 12px;border-radius:99px;border:.5px solid var(--line2);background:transparent;color:var(--muted);cursor:pointer}
.tab.on{background:var(--text);color:var(--bg);border-color:var(--text)}
.md{font-size:13.5px}
.md h1,.md h2,.md h3{font-size:15px;font-weight:600;margin:1.1em 0 .4em}
.md code{font-family:ui-monospace,Menlo,monospace;font-size:.9em;background:rgba(127,127,127,.14);padding:1px 5px;border-radius:5px}
.md pre{background:rgba(127,127,127,.1);padding:10px;border-radius:8px;overflow:auto}
.md pre code{background:none;padding:0}
.md table{border-collapse:collapse;font-size:12.5px}
.md td,.md th{border:.5px solid var(--line);padding:4px 8px}
.md blockquote{border-left:2px solid var(--line2);margin:.6em 0;padding:.1em 0 .1em 12px;color:var(--muted)}
pre.src{margin:0;border-radius:8px;font-size:12.5px;max-height:60vh;overflow:auto}
.empty{color:var(--hint);font-size:13px}
.foot{font-size:11.5px;color:var(--hint);margin-top:2rem}
</style>
</head>
<body>
<div class="wrap">
  <header class="head">
    <h1>glassbox · VLM component map</h1>
    <div class="sub">two tracks: <code>llm/</code> language backbone + <code>cv/</code> vision tower · join: <code>cv/vl</code> + <code>cv/mrope</code></div>
    <div class="seg" id="seg"></div>
    <div class="cnt" id="cnt"></div>
    <div class="legend">
      <span><i class="dot" style="background:var(--done)"></i>已实现 · 有实验/讲解</span>
      <span><i class="dot" style="background:var(--core)"></i>已实现 · core code</span>
      <span><i class="dot" style="background:var(--spec)"></i>未实现 · spec only</span>
    </div>
  </header>
  <main class="zones" id="zones"></main>
  <aside class="detail" id="detail"><div class="empty">点任一构件卡 · 看它的真实笔记与代码</div></aside>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>
const DATA = __DATA__;
const MC = {explained:'var(--done)', core:'var(--core)', spec:'var(--spec)'};
const ML = {explained:'已实现 · 有实验/讲解', core:'已实现 · core code', spec:'未实现 · spec only'};
const byKey = Object.fromEntries(DATA.components.map(c=>[c.key,c]));

function renderZones(){
  const host = document.getElementById('zones');
  for(const [z,label,desc] of DATA.zones){
    const items = DATA.components.filter(c=>c.zone===z);
    if(!items.length) continue;
    const sec = document.createElement('section'); sec.className='zone';
    sec.innerHTML = '<div class="zh"><b>'+label+'</b><span>'+desc+'</span></div>';
    const grid = document.createElement('div'); grid.className='grid';
    for(const c of items){
      const el = document.createElement('div'); el.className='card'; el.dataset.key=c.key;
      el.innerHTML = '<div class="top"><span class="nm">'+c.key+'</span>'+
        '<i class="dot" style="background:'+MC[c.mastery]+'"></i></div>'+
        '<div class="ol">'+(c.oneliner||c.qwen||'')+'</div>';
      el.onclick=()=>select(c.key);
      grid.appendChild(el);
    }
    sec.appendChild(grid); host.appendChild(sec);
  }
}

function renderProgress(){
  const cs=DATA.components, n=cs.length;
  const k={explained:0,core:0,spec:0};
  cs.forEach(c=>k[c.mastery]++);
  document.getElementById('seg').innerHTML=['explained','core','spec']
    .map(m=>k[m]?'<span style="width:'+(k[m]/n*100).toFixed(1)+'%;background:'+MC[m]+'"></span>':'').join('');
  document.getElementById('cnt').textContent=
    '共 '+n+' 个构件 · 已实现 '+(k.explained+k.core)+' · 未实现/spec '+k.spec+
    ' · join 核心：cv/vl + cv/mrope';
}

let tab='notes';
function select(key){
  document.querySelectorAll('.card').forEach(e=>e.classList.toggle('sel',e.dataset.key===key));
  const c=byKey[key], d=document.getElementById('detail');
  const notes = c.readme ? marked.parse(c.readme) : '<p class="empty">这一格还没有 README</p>';
  const codeHtml = c.code
    ? '<pre class="src"><code class="language-python">'+escapeHtml(c.code)+'</code></pre>'
    : '<p class="empty">这一格还没有实现代码（spec 阶段）</p>';
  d.innerHTML =
    '<h2>'+c.key+'</h2>'+
    '<div class="meta"><i class="dot" style="background:'+MC[c.mastery]+'"></i>'+ML[c.mastery]+
      ' &nbsp;·&nbsp; role：'+(c.qwen||'—')+'</div>'+
    (c.replaces?'<div class="row"><span>取代</span><span>'+c.replaces+'</span></div>':'')+
    (c.who?'<div class="row"><span>在用</span><span>'+c.who+'</span></div>':'')+
    (c.code_file?'<div class="row"><span>代码</span><span>'+c.key+'/'+c.code_file+'</span></div>':'')+
    '<div class="tabs"><button class="tab" data-t="notes">README</button>'+
      '<button class="tab" data-t="code">代码</button></div>'+
    '<div id="pane"></div>';
  d.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{tab=b.dataset.t;paint(notes,codeHtml);});
  paint(notes,codeHtml);
  if(window.innerWidth<880) d.scrollIntoView({behavior:'smooth'});
}
function paint(notes,codeHtml){
  document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t===tab));
  const pane=document.getElementById('pane');
  pane.className = tab==='notes'?'md':'';
  pane.innerHTML = tab==='notes'?notes:codeHtml;
  if(tab==='code') pane.querySelectorAll('code').forEach(b=>hljs.highlightElement(b));
}
function escapeHtml(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

renderProgress(); renderZones();
</script>
<div class="wrap"><div class="foot">generated from cv/ and llm/ only · rerun <code>.venv/bin/python docs/build.py</code> after component changes</div></div>
</body>
</html>"""


if __name__ == "__main__":
    main()
