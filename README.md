# 魔法画框 moving_picture

给孩子的美术作品拍照（界面和旁白是英文），在浏览器里认出画中的角色，让它按最合适的方式动起来，导出 1–3 秒无缝循环的 GIF。动画只让原作变形，不重画，所以笔触和颜色都是孩子自己的。

## 运行（macOS / Windows 相同）

1. 安装 [uv](https://docs.astral.sh/uv/)。
2. 在项目根目录放一个 `.env`（已被 git 忽略）：

   ```
   CLAUDE_API_KEY=sk-ant-...
   ```

   也可以直接设置环境变量 `ANTHROPIC_API_KEY`。
3. 启动后会自动打开 http://localhost:8765：

   ```
   uv run moving-picture
   ```

   在终端按 Ctrl+C 停止。

页面需要联网：要调用 Claude API，GIF 编码库（gifenc）和字体也从 CDN 加载。照片可以拖进页面、点 "Choose a photo"（支持 iPhone 的 HEIC），也可以用电脑摄像头拍。

## 流程

| 步骤 | 位置 | 做什么 |
|---|---|---|
| 扫描 | `vision.straighten` | 处理 EXIF 方向，把图缩到长边 1600；如果能看到纸张清晰的四条边，就做透视拉正 |
| 理解 | `analyze.ask_claude` | Claude（`claude-opus-5-5`）返回：作品范围、最多 3 个角色（范围框 + 关节树）、3 套动作方案、标题和旁白 |
| 抠图 | `vision.character_mask` | GrabCut：范围框外视为背景，骨骼经过的线视为一定属于角色 |
| 补背景 | `vision.fill_behind` | 修补角色原来位置的背景，角色动起来时不会露出空洞 |
| 绑定 | `rig.build_rig` | 在角色区域内生成三角网格，按"沿身体内部到每根骨头的距离"计算蒙皮权重 |
| 动画 | `web/puppet.js` | 每根骨头绕父关节做正弦摆动，加上整体的跳、漂、摇、呼吸；所有周期都是整数，所以首尾无缝；WebGL 渲染 |
| 导出 | `puppet.exportGif` | 长边 480px，每帧 60ms，整段动画共用一个调色板 |

关节放得不对时，大人可以点 "🔧 Fix joints" 拖动关节点，然后重新抠图和绑定。

点 "🎬 Make a Grok video" 会下载裁好的原作图片，并把 Claude 为当前动作写好的英文视频提示词复制到剪贴板（页面上也会显示），然后在 [Grok Imagine](https://grok.com/imagine) 里用 SuperGrok 订阅手动上传图片、粘贴提示词、生成视频。订阅不含 API 额度，所以这一步不自动调用 Grok。

## 测试

```
uv run pytest -q
```

测试用 `tests/synth.py` 生成的合成"桌上的画"，不调用 Claude API。

## 已知局限

- 角色上面压着别的东西（比如贴在它身上的圆片、横条）时，那些东西会被一起抠进去，跟着角色动。
- 补出来的背景是模糊色块。背景越花，角色整体跳动时越容易看出来（提示词里已经让 Claude 在这种情况下少用整体跳动）。
- 一张图要调用一次 Claude，大约需要 15–20 秒；从拖入照片到开始动约 20 秒。
- 色彩丰富的作品导出的 GIF 大约 2MB。
