# GitHub 上传指南

## 📋 前提条件

### 1. 安装 Git

如果系统中未安装 Git，请访问 https://git-scm.com/download/win 下载并安装。

安装完成后，打开新的 PowerShell 终端验证安装：
```powershell
git --version
```

### 2. 在 GitHub 创建仓库

1. 登录 GitHub: https://github.com
2. 点击右上角 **+** → **New repository**
3. 仓库名称：`faers-sglt2-dka`
4. 类型：**Public** (公开) 或 **Private** (私有)
5. **不要**勾选"Add a README file"（我们已有 README）
6. 点击 **Create repository**

---

## 🚀 上传步骤

打开 PowerShell，依次执行以下命令：

```powershell
# 1. 进入项目目录
cd E:\FAERS_DKA\faers-sglt2-dka-github

# 2. 初始化 Git 仓库
git init

# 3. 配置 Git 用户信息（只需执行一次）
git config --global user.name "huaibing-xtu"
git config --global user.email "huaibing@xtu.edu.cn"

# 4. 添加所有文件到暂存区
git add .

# 5. 创建初始提交
git commit -m "Initial commit: FAERS SGLT2-DKA Analysis

- Complete pharmacovigilance signal detection pipeline
- Explainable machine learning for DKA report identification
- SHAP interpretability and temporal validation"

# 6. 重命名主分支为 main
git branch -M main

# 7. 添加远程仓库
git remote add origin https://github.com/huaibing-xtu/faers-sglt2-dka.git

# 8. 推送到 GitHub
git push -u origin main
```

---

## ⚠️ 常见问题

### 问题 1：Git 未识别

**解决**：安装 Git 或检查 PATH 环境变量。

### 问题 2：推送时被要求输入密码

**解决**：建议使用 GitHub Personal Access Token 代替密码：
1. 访问 https://github.com/settings/tokens
2. 生成新 Token（选择 `repo` 权限）
3. 使用 Token 代替密码进行认证

### 问题 3：已有其他 Git 仓库

**解决**：如果目录中已有其他 Git 仓库，先备份或删除 `.git` 文件夹。

---

## ✅ 上传后检查

上传完成后，访问 https://github.com/huaibing-xtu/faers-sglt2-dka 确认：
- 所有文件已显示
- README.md 正确渲染
- 分支为 `main`

---

## 📝 已完成的工作

- ✅ 更新 README.md 中的仓库链接为 huaibing-xtu/faers-sglt2-dka
- ✅ 更新 QUICKSTART.md 中的克隆链接
- ✅ 更新 CITATION.cff 中的引用链接
- ✅ 更新 setup.py 中的仓库 URL
- ✅ 更新 CONTRIBUTING.md 中的克隆命令
- ✅ 更新 CI 工作流中的链接
- ✅ 更新联系方式为 huaibing@xtu.edu.cn

---

## 🔄 后续更新

如需更新代码：
```powershell
cd E:\FAERS_DKA\faers-sglt2-dka-github
git add .
git commit -m "描述你的更改"
git push
```

---

*Last updated: 2026-06-10*
