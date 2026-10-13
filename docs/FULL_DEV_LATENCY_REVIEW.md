# 完整 dev 配对延迟扩展源码审查

2026-10-14，fresh reviewer `/root/review_full_dev_latency` 实际读取diff与接口源码。
SOURCE_ONLY PASS，真正源码阻断0；same-family/provisional，requested gpt-6-astra/max，
actual routing/effort UNATTESTED。不把SOURCE PASS解释为真实NN已通过。
Git基点7189c12806ca6354c5a62b0d116b6b08a685eb15；目标源码SHA
`b889fc86fe315357369472c6bb0bb18d3159d40ce9928871ddc44de505e1b43b`。
相对旧前缀脚本只13行新增/8行删除，封存17项实际17/17一致；该脚本不在seal内。

full-dev使用全部manifest dev，从每序列frame0顺序执行全部rgb帧；断言RGB/TIR/boxes同长。
不根据后续GT有效性跳帧，GT只用于首框，两侧独立搜索、模板状态与解码。
原CADTrack仍strict加载原权重，无精度评分、baseline框落盘、反向或优化器更新。
wrapper在Tracker导入前设置crop alias，不改变原CADTrack的sample_target。
初始化/解码/结果提取及结束CUDA同步均计入，加载排除；warm只事后取frame>=9。
预期full54646、warm53773、非初始化54549/侧，必须运行完成后核实，不能写为实测结果。

reviewer本地未找到manifest，未独立核验其具体序列/首框；读取旧完整范围报告与实际源码。
结果JSON源列表不含wrapper/三个crop helper，因此正式启动凭据必须记录真实入口命令并绑定17项seal；
单份延迟JSON不足以证明实际crop实现。常驻双模型、缓存、交替先后均是测量限制。
裁剪/模板更新/定位路径有差别，总差值属于系统对照，不能归因于一个模块或冒充完整test速度。

源码侧实际检查：reviewer Python3.7.6 AST PASS、git diff --check PASS、源hash核对。
执行者随后在原conda CUDA隐藏CLI：原生PID705573，00:10:30.671→00:10:34.526 BJT，
真实exit0，stderr空，--help输出包含--full-dev。记录 `reports/full_dev_latency_cli_native.json`。
仅CLI导入/解析完成；真实模型加载、前向、完整配对延迟仍NOT_RUN。
