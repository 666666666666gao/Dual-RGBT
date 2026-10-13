# 环境和可复现命令

环境已从现有pytracking conda环境通过conda-pack迁至
`/home/gaob/Dual-RGBT/conda-envs/dual_rgbt`。后续运行使用此目录内解释器。
原包缓存部分缺失；打包日志保留该警告，验收以实际完整模型CUDA执行为准。
`conda-unpack`已用该环境自身解释器执行，不依赖PATH中的python。

GPU0真实M0通过，报告为`reports/sanity_gpu0.json`：Python前缀指向上述目录、
PyTorch1.12.1+cu113，十种执行组合均完成真实更新与严格重载，峰值约3GB。
此处不是精度或实时延迟结果。训练只用GPU0/1。

独立文档复现命令（GPU1，运行约20–60秒；原GPU0报告已单独保留）：

```powershell
ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes 2025 'cd /home/gaob/Dual-RGBT; CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 conda-envs/dual_rgbt/bin/python scripts/sanity.py > reports/sanity_doc_gpu1.log 2>&1'
```

退出0且`reports/sanity.json`状态PASS才可宣称命令复现通过。不要替换命令，
不要修改源码或环境，不要运行baseline test、不启动阶段A，不读取温度功率。

阶段A（M0和文档复现通过后，由主代理单次启动）：

```bash
cd /home/gaob/Dual-RGBT
bash scripts/run_stage_a.sh
```

前台命令应在独立的持久进程中运行，日志保存到项目`runs/`。首次短程是两轮、
每轮2000样本、1000次AdamW更新，重新从用户原权重开始，不从M0十步状态开始。
