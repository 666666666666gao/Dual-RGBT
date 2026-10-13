# 实际问题记录

| 编号 | 现象与证据 | 处理 | 状态 |
|---|---|---|---|
| E01 | GitHub直连schannel TLS失败；7897无监听；7898实际可用 | 此仓库使用现有SOCKS代理7898，未改全局Git配置 | ls-remote和clone成功 |
| E02 | 原权重torch.load报缺少lib.train.admin.local | 提供实际序列化所需EnvironmentSettings类，路径指向本项目 | CPU读取成功，epoch14，103142866 state元素 |
| E03 | conda离线clone需已不存在的包缓存，失败于EnforceUnusedAdapter | 使用conda-pack迁入项目环境；保留conda_create.log | 迁移及GPU见证待完成 |
| E04 | conda-pack包没有__main__入口，首次命令失败 | 按API调用conda_pack.pack，原日志保留 | 第二次打包进行中 |

当前没有新方法精度、辅助收益或速度结果。不能把这些环境问题解释为方法失效。
