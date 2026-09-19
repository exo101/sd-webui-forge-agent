---
name: archify
description: 把 WebUI 插件结构、项目结构、工作链路（工作流 / API 调用时序 / 数据流 / 状态生命周期）渲染成可交互的自包含 HTML 图表（内联 SVG，支持缩放、深浅主题、导出 PNG/JPEG/WebP/SVG/WebM）。支持 5 种图：architecture（组件/服务/边界/基础设施）、workflow（流程/审批/工具调用/CI-CD）、sequence（API 调用链/请求生命周期/异步时序）、dataflow（管道/ETL/血缘/消费者）、lifecycle（状态机/重试/等待与终止态）。当用户要求"看懂/画出来/可视化 插件结构、项目结构、工作链路"，或要把 Mermaid flowchart/sequenceDiagram/stateDiagram 转成更美观的图表时使用。本技能内置在 WebUI 智能体扩展中，通过 generate_diagram / validate_diagram 工具完成渲染，无需 shell 访问。
license: MIT
metadata:
  version: "2.17"
  author: tt-a1i
  based_on: Cocoon-AI/architecture-diagram-generator (MIT, v1.0)
  distribution: webui-agent-builtin
---

# Archify（内置技能）

把一段小的、带类型的 JSON 规格渲染成自包含、可交互的 HTML 图表。
静态输出是默认行为；仅当用户明确要求演示/汇报时才启用动效。

运行时位置（扩展内置，**无需安装任何依赖**）：

```
<EXT>/runtime/archify/
  schemas/<type>.schema.json + common.schema.json   字段契约
  examples/<type>-*.json                            字段形状参考（不是事实来源）
  references/authoring-contract.md                  字段枚举/间距/几何修复规则（按需读）
  references/delivery-contract.md                   交付回执字段（按需读）
```

其中 `<EXT>` 是 sd-webui-agent-dev-2.4 扩展根目录。
用 `read_workspace_file` 读取相对路径即可（如 `webui/extensions/sd-webui-agent-dev-2.4/runtime/archify/schemas/workflow.schema.json`）。
渲染一律通过智能体工具 `validate_diagram` / `generate_diagram` 完成，**不要**让用户去 shell 里手动跑 node。

## 快速创作路径（Fast authoring path）

1. **选类型**：从 5 种中挑一个最贴合问题的（见下方 Type router）。
2. **读契约**：用 `read_workspace_file` 读取三样东西——
   对应类型的 `runtime/archify/schemas/<type>.schema.json`、
   `runtime/archify/schemas/common.schema.json`、
   以及 `runtime/archify/examples/` 下同类型的一个 JSON 示例。
   只读这三个文件。示例只用来学习字段形状，不要照抄其中的事实；新图用新的稳定 ID 和贴合本项目的措辞。
   新建 workflow 用 `schema_version: 2`；只有要保留已有 workflow 的固定几何时才用 `schema_version: 1`。
3. **先写规格**：下一个动作必须写出候选 JSON 规格（用 `repair_workspace_file` 写入
   `webui/extensions/sd-webui-agent-dev-2.4/outputs/diagrams/<名字>.json`，或交给 `generate_diagram` 的 `spec` 字段内联）。
   不要先用文字规划精确坐标。从一条清晰的主路径、短小的旁支、稀疏的标签开始，主节点**最多 12 个**。
   `meta.quality_profile` 设为 `"showcase"`（除非用户明确要高密度的 `standard`）。
   先走自动布线；在诊断明确要求前，不要添加 `via` / `channelX` / `channelY` / `labelAt`。
   每轮修复最多应用 1 个被诊断出的几何控制项。
4. **每次编辑后 validate**：调用 `validate_diagram(type, spec)`。
   showcase 通过要求：9 项 artifact check 全部报告、0 个 composition error、0 个 warning。
   如果规格漏写或拼错 `meta.quality_profile`，先修这个字段再动几何。
   workflow v2 的几何诊断看 validate 返回的编译回执；solver 内部不是创作控制项。
   最终一次 validate 通过后，规格即冻结：之后不要再改它。
5. **交付**：调用 `generate_diagram(type, spec)` 产出 HTML。
   工具内部会做交付校验（deliver）；非零退出绝不能描述成成功。
   交付失败会保留上一次的旧产物，所以不要在失败路径上基于旧 HTML 下结论。
   验证失败时：只改被诊断出的 `subject`，依据 `evidence`，从 `supportedFixes` 里选修复，然后重跑。
   当连续两轮都无法把错误数降到新的最小值时，停止并如实报告未解决的诊断。

## Type router

| 类型 | 用于 |
|---|---|
| `architecture` | 组件、服务、云/安全边界、基础设施、插件模块关系 |
| `workflow` | 流程、审批门、工具调用链、runbook、CI/CD、智能体工作链路 |
| `sequence` | API 调用链、请求生命周期、异步 trace、返回路径 |
| `dataflow` | 管道、ETL/ELT、数据血缘、治理、消费者 |
| `lifecycle` | 状态/状态转移、重试、等待与终止态 |

拿不准时，优先读该类型 schema + 示例再判断；场景示例是结构参考，不是要抄的事实。

## Mermaid 输入

用户贴 Mermaid 时：读 Mermaid 拿拓扑与语义，然后**新写** Archify JSON；不要机械搬 Mermaid 样式。

- `flowchart` / `graph` → `workflow`；组件图 → `architecture`。
- `sequenceDiagram` → `sequence`；参与者变成语义参与者，箭头变成消息。
- `stateDiagram` → `lifecycle`；保留状态与转移的语义，不保留 Mermaid 样式。

## 创作不变量（Authoring invariants，摘要）

- 一条显而易见的主路径；旁支从最近的主路径节点分出。低价值边先删，再谈路由控制。
- 默认省略 `meta.visual_preset`（经典风格）；只有用户明确要 `signal-flow` / `blueprint` / `editorial` 才设置。
- 默认省略 `meta.subtitle`；绝不编造复述标题/节点的副标题。
- 把桌面查看器当首屏产物；为笔记本与外接显示器生成一份响应式产物，不要为特定设备另做 HTML。
  不要靠 `overflow: hidden`、裁切内容、图内滚动条、拉高 SVG、缩小字体来"假装"通过。
- 默认省略 `meta.legend`（自动）；需要时用 `mode: auto|all|hidden`，标签不改语义。
- `meta.locale` 只控制查看器 UI：`"en"` 或 `"zh-CN"`；其他语言省略并明确告知用户查看器 UI 回退为英文。
  渲染器从不翻译你写的内容。
- 精确保留产品名、代码标识符、命令、协议、API 路径、环境变量名（可保留英文）。
- 组件类型：`frontend`、`backend`、`database`、`cloud`、`security`、`messagebus`、`external`；
  变体：`default`、`emphasis`、`security`、`dashed`。
- 关系标签是语义数据。标签冲突时：移动标签 → 调整路由/间距 → 在保住语义的前提下缩短措辞。
  删除标签不是几何修复；只有当两端点完全蕴含该信息且不含协议/动作/方向/同步异步/跨边界机制时才可省略。
- 默认省略 `meta.engineering_profile`；仅当用户明确要生产部署拓扑/所有权移交/失效封闭评审且事实已知时，
  才启用 `deployment-ownership`。启用后不能为了过验证而删掉它。
- 间距指"清晰空隙"而非中心距离：关系标签的空隙必须大于其测量掩码宽度。
- 自动路由拥有端点侧：首末段必须垂直于该侧进出。
- 绝不接受：边穿过无关的不透明节点、歧义的共享走廊、关系标签遮住另一条路由。

需要字段枚举、间距计算、几何修复规则、模式特定布局时，再读
`runtime/archify/references/authoring-contract.md`。

## 交付（Delivery）

- 修复期用 `validate_diagram`，最终一次用 `generate_diagram`。
- `generate_diagram` 会冻结规格字节、渲染并检查、原子提交 HTML，并回执 SHA-256 与字节数。
  这是确定性产物证据，不等于在浏览器里跑过查看器。
- 感知层面的"好不好看"需要真实的人或具备图像能力的评审，不要声称你做了没做的视觉审查。
- `generate_diagram` 返回 HTML 的绝对路径。Gradio 聊天界面不能内嵌渲染 HTML，
  所以把该路径告诉用户，请用户在浏览器中打开查看。

## 回退（Fallback）

- 不要运行联网的更新检查或品牌抓取；本技能是离线内置副本。
- 若 `generate_diagram` 报错提示 Node 运行时不可用，如实报告，并建议检查整合包内是否包含 Node（archify 需要 `node` 在 PATH 中）。
- 极少数情况（工具链不可用且用户仍要产物）：把架构 SVG 手工放进 `runtime/archify/assets/template.html`，
  用 CSS 语义类而非内联颜色，遵循 `references/delivery-contract.md` 的视觉审查契约。

## 输出（Output）

返回：受检 HTML 路径、图表类型、验证摘要、规格/产物回执、浏览器证据状态、真实的视觉审查状态。
非零命令不能声称成功；没做的视觉检查不能声称已做。
