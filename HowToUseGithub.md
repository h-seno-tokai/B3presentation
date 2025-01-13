以下は、`HowToUseGithub.md` に記載する内容です。これにより、Git の基本的な使い方がわかります。

---

# How to Use GitHub

このドキュメントでは、Gitの基本操作からGitHubでのプルリクエスト作成までの手順を説明します。

---

## **1. Git リポジトリの初期化**

1. リポジトリを初期化します：
   ```bash
   git init
   ```

2. Git の状態を確認します：
   ```bash
   git status
   ```

---

## **2. ステージングとコミット**

1. ファイルをステージングします（例：全ファイルをステージング）：
   ```bash
   git add .
   ```

2. ステージングした内容を確認します：
   ```bash
   git status
   ```

3. コミットを作成します：
   ```bash
   git commit -m "コミットメッセージを記入"
   ```

---

## **3. リモートリポジトリの設定**

1. リモートリポジトリを追加します：
   ```bash
   git remote add origin git@github.com:<username>/<repository>.git
   ```

2. リモートリポジトリを確認します：
   ```bash
   git remote -v
   ```

---

## **4. ブランチの作成と切り替え**

1. 新しいブランチを作成します：
   ```bash
   git checkout -b <branch-name>
   ```

2. 作成したブランチをリモートリポジトリにプッシュします：
   ```bash
   git push -u origin <branch-name>
   ```

---

## **5. GitHub でのプルリクエスト作成**

1. ブラウザでリモートリポジトリにアクセスします。
2. 「Compare & pull request」ボタンをクリックします。
3. 以下を記入します：
   - タイトル: 変更内容を簡潔に記載。
   - 説明: 変更の背景や詳細を記入。

---

## **6. その他の便利なコマンド**

- **変更内容の確認**:
   ```bash
   git diff
   ```

- **ブランチの一覧を表示**:
   ```bash
   git branch
   ```

- **リモートリポジトリとローカルの同期**:
   ```bash
   git pull origin <branch-name>
   ```

- **不要なブランチを削除**:
   ```bash
   git branch -d <branch-name>
   ```

---

