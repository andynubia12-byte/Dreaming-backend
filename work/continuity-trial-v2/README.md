# continuity-trial-v2（空数据开发副本）

完整保留当前 Python 源码、工作台与旧试用前端、CLI；历史数据和凭据未迁入。

在项目根执行 `./scripts/start-local.sh`，默认工作台端口 8794；端口占用时使用 `--port 8794`。
旧试用入口增加 `--trial`。仅依赖 Python 标准库；启动不会调用供应商。

维护依赖是旧 v2.6 + citation-alias wire，不属于 v1 选定摘要方法；没有接通 8788 写入。
运行、认证、数据位置与新前端 API 见 [运行与前端接入](../../docs/运行与前端接入.md)。
复制来源与检查记录见 [打包报告](../../packaging-code-report.json)。
