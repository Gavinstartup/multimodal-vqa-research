# Git 提交规范

- **禁止在 commit message 中添加任何 AI 署名 trailer**，包括但不限于
  `Co-Authored-By: Claude <noreply@anthropic.com>`、
  `Co-Authored-By: Claude Opus 5 ... <noreply@anthropic.com>`、
  `Generated with Claude Code` 等。
  原因：GitHub 会解析 `Co-Authored-By` trailer 并把对应账号计入 contributor
  列表和贡献图，这不是本项目希望的署名方式。
- 同理，PR 描述中也不要添加 `🤖 Generated with [Claude Code]` 之类的尾注。
- commit message 只写变更本身：一行主题 + 必要时空行后的正文说明「为什么」。

这条规则优先于 Claude Code 的默认署名行为。
