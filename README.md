# ZIWIGPT 井场工况预警决策台

面向油气工程现场的 Streamlit POC，用于维泰能源集团（Vertechs Group）解决方案工程师面试演示。

## 一键启动

```powershell
# 在本目录执行
C:\Users\13773\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m streamlit run app.py --server.port 8501
```

如果系统 `python` 正常，也可以：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

浏览器打开 `http://127.0.0.1:8501`。

## 演示路径

1. 左侧选择任一案例（四类工况矩阵）。
2. 点击「运行诊断 →」。
3. 观察 SPP / ROP / 水力负荷 / 安全余量卡片。
4. 阅读 DeepSeek 实时生成的双语诊断。
5. 点击「📥 导出 A4 双语决策签批报告 (PDF)」。

### 四类工况矩阵

| 标签 | 井 / 场景 | 数据文件 |
|---|---|---|
| RED | 川西-03井 · 环空憋堵 / 卡钻前兆 | `sample_ddr.txt` |
| GREEN | 塔里木-01井 · 正常钻进 | `sample_tarim_01.txt` |
| ORANGE | 渤海-19井 · 压力陡降 / 疑似漏失前兆 | `sample_bohai_leak.txt` |
| YELLOW | 沙特-Ghawar井 · 致密碳酸盐岩硬夹层 | `sample_saudi_hard.txt` |

也可上传新的 `.txt` 日报，格式需包含 `[WELL]`、`[BASELINE]`、`[CURRENT]` 段。

## 架构

- `check_hydraulics.py`：确定性规则引擎，提供 `loadddr` / `analyzedata`。
- `app.py`：Streamlit UI + DeepSeek 流式调用 + 案例矩阵分级 + 离线专家兜底 + ReportLab PDF。
- `.streamlit/secrets.toml`：仅用于本地演示的 API Key（已被 `.gitignore` 排除，不会上传）。
- `sample_ddr.txt`：川西-03 高危异常案例。
- `sample_tarim_01.txt`：塔里木-01 正常工况案例。
- `sample_bohai_leak.txt`：渤海-19 裂缝性砂岩疑似漏失案例。
- `sample_saudi_hard.txt`：沙特-Ghawar 致密碳酸盐岩硬夹层案例。

## 云端部署与 Key 优先级

云端（Streamlit Community Cloud 或其他平台）不需要上传 `.streamlit/secrets.toml`，在平台 Secrets 面板配置：

```toml
DEEPSEEK_API_KEY = "sk-..."
```

`app.py` 的解析顺序：

1. `st.secrets.get("DEEPSEEK_API_KEY")`：云端 Secrets，优先级最高；
2. 左侧「临时 API Key（可选）」：仅保存在当前浏览器会话，不写入磁盘；
3. `DEEPSEEK_API_KEY` 环境变量；
4. 以上都没有：自动使用内置专家稿，演示不中断。

因此，云端没有配置 Secret 时，可以直接在界面粘贴 Key 完成演示；本地 `.streamlit/secrets.toml` 只服务本机运行，不会进入 Git。

## AI 模型说明

默认模型为 `deepseek-flash`，每次点击「运行诊断」都会直接请求该模型，不再进行模型级重试或降级切换。侧边栏引擎状态直接显示 `Engine: deepseek-flash (Connected)`，UI 与 PDF 均记录该模型。调用使用标准库 HTTPS，不依赖 OpenAI SDK。

为确保 10 秒级演示响应，请求中设置：

```json
{"thinking": {"type": "disabled"}}
```

如果网络超时、Key 缺失、请求失败或返回空正文，应用会自动使用内置高保真专家稿，演示不会崩溃；这不属于模型降级，而是离线安全兜底。

## 安全边界

- 所有案例均为合成数据，不是真实井史。
- SPP/ROP 数值只由规则引擎计算，LLM 不参与加减乘除。
- 报告是 POC 决策辅助，不替代现场工程师、钻井设计和井控程序。
- 云端部署使用平台 Secrets；本地 `.streamlit/secrets.toml` 已被 `.gitignore` 排除。
- 界面输入的 Key 只存在 Streamlit 会话内，不写磁盘；面试结束后建议轮换该 Key。
