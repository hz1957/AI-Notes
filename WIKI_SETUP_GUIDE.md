# 搭建个人知识库 Wiki 并部署到 GitHub Pages 指南

本文档将指导你使用 [VitePress](https://vitepress.dev/) 搭建一个现代化、极简且美观的文档/Wiki 风格网站，最终将其自动部署到 GitHub Pages。

**为什么选择 VitePress?**
- **由于 Vue 驱动**：极速的开发服务器和构建速度。
- **Markdown 中心**：专注于写作，支持 GitHub 风格的表格、代码高亮等。
- **美观的默认主题**：包含夜间模式、侧边栏、搜索功能，非常适合知识库。

---

## 步骤 1：环境准备

确保你的电脑上已经安装了 **Node.js** (推荐 v18 或更高版本)。
在终端输入以下命令检查：
```bash
node -v
```

---

## 步骤 2：初始化项目

1. 打开你的项目目录 (例如 `/Users/haoming.zhang/PyCharmMiscProject/AI-notes`)。
2. 打开终端，运行以下命令初始化项目结构：

```bash
# 安装 VitePress
npm add -D vitepress

# 启动设置向导
npx vitepress init
```

3. **设置向导选项建议**：
   - **Where should VitePress initialize the config?**: 直接回车 (默认 `./`)
   - **Site title**: `我的知识库` (或者你喜欢的名字)
   - **Site description**: `个人学习笔记与知识沉淀`
   - **Theme**: `Default Theme`
   - **Use TypeScript for config and theme files?**: `Yes` (推荐)
   - **Add VitePress npm scripts to package.json?**: `Yes`

安装完成后，你的目录结构大致如下：
```
.
├─ docs/                    # 文档源文件目录
│  ├─ .vitepress/           # 配置文件目录
│  │  └─ config.mts         # 网站配置文件
│  ├─ api-examples.md
│  ├─ markdown-examples.md
│  └─ index.md              # 首页
├─ node_modules/
├─ package.json
└─ package-lock.json
```

---

## 步骤 3：本地预览

在终端运行以下命令启动本地服务器：
```bash
npm run docs:dev
```
浏览器打开 `http://localhost:5173` 即可看到初始好的网站。

---

## 步骤 4：配置侧边栏与导航 (搭建 Wiki 骨架)

打开 `docs/.vitepress/config.mts`，这里是控制网站外观的核心。
你可以参照以下配置修改，构建知识库的分类：

```typescript
import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "我的知识库",
  description: "Personal Knowledge Wiki",
  
  // 部署到 GitHub Pages 的仓库名 (如果是 username.github.io 形式则不需要 base，否则需要设置 /repo-name/)
  // base: '/AI-notes/', 

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: 'Python', link: '/python/intro' },
      { text: 'AI 笔记', link: '/ai/intro' }
    ],

    sidebar: {
      // 当用户在 /python/ 目录下时显示此侧边栏
      '/python/': [
        {
          text: 'Python 学习',
          items: [
            { text: '简介', link: '/python/intro' },
            { text: '基础语法', link: '/python/basic' },
            { text: '高级特性', link: '/python/advanced' }
          ]
        }
      ],
      // 当用户在 /ai/ 目录下时显示此侧边栏
      '/ai/': [
        {
          text: '人工智能',
          items: [
            { text: 'LLM 简介', link: '/ai/intro' },
            { text: 'Prompt Engineering', link: '/ai/prompt' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME' }
    ],
    
    // 启用本地搜索
    search: {
      provider: 'local'
    }
  }
})
```

**创建对应的文档文件**：
需要在 `docs` 目录下新建 `python` 和 `ai` 文件夹，并创建相应的 `.md` 文件 (如 `docs/python/intro.md`)。

---

## 步骤 5：部署到 GitHub Pages

我们将使用 **GitHub Actions** 实现当你 `git push` 时自动更新网页。

### 1. 准备 GitHub 仓库
1. 在 GitHub 上新建一个仓库（例如命名为 `AI-notes`）。
2. 确保你的 `package.json` 中 `"scripts"` 包含 `"docs:build": "vitepress build docs"` (向导自动生成的一般都有)。
3. 确保 `.gitignore` 忽略了 `node_modules` 和 `.vitepress/dist`。

### 2. 创建 GitHub Actions Workflow
在项目根目录下新建目录和文件： `.github/workflows/deploy.yml`

**文件内容如下：**

```yaml
# .github/workflows/deploy.yml
name: Deploy VitePress site to Pages

on:
  # 在推送到 main 分支时触发
  push:
    branches: [main]

  # 允许手动触发
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  # 构建任务
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0 # 如果未启用 lastUpdated，则不需要

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Setup Pages
        uses: actions/configure-pages@v5

      - name: Install dependencies
        run: npm ci

      - name: Build with VitePress
        run: npm run docs:build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: docs/.vitepress/dist

  # 部署任务
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    name: Deploy
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 3. 配置 GitHub 仓库设置
1. 将本地代码推送到 GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/你的用户名/你的仓库名.git
   git push -u origin main
   ```
2. 打开 GitHub 仓库页面 -> **Settings** -> **Pages**。
3. 在 **Build and deployment** 部分，Source 选择 **GitHub Actions**。
4. 此时 Actions 应该会自动开始构建。等待完成后，你就可以在 settings 页面上方看到你的网站链接 (例如 `https://yourname.github.io/AI-notes/`)。

> **注意**：如果你的仓库不是 `username.github.io`，那么你的网站路径会有一个子路径 (sub-path)。请务必在 `docs/.vitepress/config.mts` 中设置 `base` 属性，例如 `base: '/AI-notes/'`。

---

## 常用操作

- **启动本地服务**: `npm run docs:dev`
- **编写内容**: 直接在 `docs` 目录下添加 Markdown 文件。
- **添加侧边栏**: 修改 `.vitepress/config.mts`。
- **发布更新**: `git add .`, `git commit`, `git push`。
