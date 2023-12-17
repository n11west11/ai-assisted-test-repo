<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>AI-ASSISTED-TEST-REPO</h1>
<h3>◦ Unleashing AI Power, One Test at a Time!</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Poetry-60A5FA.svg?style=flat-square&logo=Poetry&logoColor=white" alt="Poetry" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Playwright-2EAD33.svg?style=flat-square&logo=Playwright&logoColor=white" alt="Playwright" />
<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat-square&logo=JSON&logoColor=white" alt="JSON" />
</p>
<img src="https://img.shields.io/github/license/n11west11/ai-assisted-test-repo?style=flat-square&color=5D6D7E" alt="GitHub license" />
<img src="https://img.shields.io/github/last-commit/n11west11/ai-assisted-test-repo?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/n11west11/ai-assisted-test-repo?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/n11west11/ai-assisted-test-repo?style=flat-square&color=5D6D7E" alt="GitHub top language" />
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📦 Features](#-features)
- [📂 repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running ai-assisted-test-repo](#-running-ai-assisted-test-repo)
    - [🧪 Tests](#-tests)
- [🛣 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

The AI-Assisted-Test-Repo is a project aimed at improving UI testing through the integration of artificial intelligence. The repository contains a Python-based UI testing assistant named Qubert. This assistant uses OpenAI’s APIs and Python Playwright mechanisms to process instructions and deliver efficient UI testing solutions. Qubert is versatile and can interpret code, retrieve data, and make function calls, all within a browser context. By safeguarding environment variables, authorization is always in place. This project provides developers consistent, robust, and smart UI testing automation, enhancing production quality and reducing manual efforts.

---

## 📦 Features

|    | Feature            | Description                                                                                                        |
|----|--------------------|--------------------------------------------------------------------------------------------------------------------|
| ⚙️ | **Architecture**   | The project is divided into a main application directory (`ai_assisted_test_repo`) and a subdirectory for cookbooks (`ai_assisted_test_repo/cookbooks/ui_assistant`). |
| 📄 | **Documentation**  | The code summaries are in-depth, but there are no standalone documentation files, like README, in the repository. |
| 🔗 | **Dependencies**   | The project utilizes libraries like pymongo, chainlit, playwright, beautifulsoup4, and readmeai. |
| 🧩 | **Modularity**     | The system appears to be modular, with separate scripts handling different parts of the UI testing process. |
| 🧪 | **Testing**        | No explicit testing strategies or tools have been identified in the current repository structure. |
| ⚡️  | **Performance**    | Performance aspects are not detailed in the repository, such as runtime efficiency or resource usage. |
| 🔐 | **Security**       | API keys are presumably managed via environment variables, aiding in security. Explicit security measures are unclear. |
| 🔀 | **Version Control**| The project is hosted on GitHub, indicating git-based version control is likely used. |
| 🔌 | **Integrations**   | The system integrates with OpenAI's APIs for creating and managing a UI testing assistant. |
| 📶 | **Scalability**    | The system's scalability capabilities are not evident from the current structure. Further codebase examination is needed. |


---


## 📂 Repository Structure

```sh
└── ai-assisted-test-repo/
    ├── ai-assisted-test-repo.code-workspace
    ├── ai_assisted_test_repo/
    │   ├── cookbooks/
    │   │   └── ui_assistant/
    ├── poetry.lock
    └── pyproject.toml

```

---


## ⚙️ Modules

<details closed><summary>Root</summary>

| File                                                                                                                                      | Summary                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---                                                                                                                                       | ---                                                                                                                                                                                                                                                                                                                                                                                                                    |
| [ai-assisted-test-repo.code-workspace](https://github.com/n11west11/ai-assisted-test-repo/blob/main/ai-assisted-test-repo.code-workspace) | The code represents a workspace configuration for Visual Studio Code, defining the structure of the project ai-assisted-test-repo. It includes three folders: a virtual environment (.venv), the main project folder (ai-assisted-test-repo), and a cookbook folder. The code also present an empty settings object. No specific workspace settings have been set.                                                     |
| [pyproject.toml](https://github.com/n11west11/ai-assisted-test-repo/blob/main/pyproject.toml)                                             | The provided code describes a Python project configuration file (pyproject.toml) within an AI-assisted test repository. The project, named ai-assisted-test-repo, primarily utilizes Python 3.10 to 3.11 and depends on libraries like pymongo, chainlit, playwright, beautifulsoup4 and readmeai. A distinct group of development dependencies include setuptools. The project uses poetry-core for its build system. |
| [poetry.lock](https://github.com/n11west11/ai-assisted-test-repo/blob/main/poetry.lock)                                                   | The code is a section of the poetry.lock file, automatically generated by Poetry, a dependency management tool for Python. The excerpt lists two packages: aiofiles and aiohttp, including their names, versions, descriptions, python version requirements, whether they're optional, and their file hashes. These aid in ensuring the consistency and integrity of the packages when used in a Python project.       |

</details>

<details closed><summary>Ui_assistant</summary>

| File                                                                                                                                                 | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---                                                                                                                                                  | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [create_assistant.py](https://github.com/n11west11/ai-assisted-test-repo/blob/main/ai_assisted_test_repo/cookbooks/ui_assistant/create_assistant.py) | The script imports necessary modules, loads the OpenAI API key and initializes the OpenAI client, dynamically loads UI tools from JSON files, maps tool function names from those JSON files to the corresponding imported functions, and defines an asynchronous create function. This create function creates a new UI testing assistant named Qubert equipped with instructions and tools from OpenAI, then saves the assistant's ID to assistants.json.                                                                           |
| [app.py](https://github.com/n11west11/ai-assisted-test-repo/blob/main/ai_assisted_test_repo/cookbooks/ui_assistant/app.py)                           | This script sets up a UI testing assistant using OpenAI's APIs and Python Playwright. It initiates a browser context, creates chat threads, and processes user or assistant messages. Messages can contain instruction texts or images. The script handles three types of tool calls: code interpretation, data retrieval, and function call. Function calls operate with browser context. Regular checks for changes in the status of a step in the running chat occur, leading to the script's termination upon certain conditions. |
| [.env.example](https://github.com/n11west11/ai-assisted-test-repo/blob/main/ai_assisted_test_repo/cookbooks/ui_assistant/.env.example)               | The provided code structure is for an AI-assisted test repository. The main folder contains settings, configurational files, and a cookbooks sub-directory housing ui_assistant. The.env.example file in ui_assistant stores environment variables for the OpenAI API key and the Assistant ID, crucial for connecting with OpenAI's services. The API key provides authorized access, and the Assistant ID identifies a specific AI assistant.                                                                                       |
| [assistants.json](https://github.com/n11west11/ai-assisted-test-repo/blob/main/ai_assisted_test_repo/cookbooks/ui_assistant/assistants.json)         | The code represents a project structure for an AI-assisted test repository. It includes a directory for cookbooks, specifically for a UI assistant named Qubert which is identified by a unique key in a JSON file. Additionally, it features files for workspace settings, Python project specifications, and dependency lock files.                                                                                                                                                                                                 |

</details>

<details closed><summary>.chainlit</summary>

| File                                                                                                                                           | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                                                                            | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [config.toml](https://github.com/n11west11/ai-assisted-test-repo/blob/main/ai_assisted_test_repo/cookbooks/ui_assistant/.chainlit/config.toml) | The code is a configuration file for an AI-assisted application named Qubert. It sets up telemetry, session timeout, user environment variables, caching options, and safety features. User interactivity options including prompts, HTML and latex processing, file uploads, and speech-to-text are configured. It also defines the UI properties like application and chatbot names, default display behaviors, collapse and expansion states of messages, UI theme colors for light and dark modes. No personal data collection is noted. |

</details>

---

## 🚀 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`- ℹ️ Dependency 1`

`- ℹ️ Dependency 2`

`- ℹ️ ...`

### 🔧 Installation

1. Clone the ai-assisted-test-repo repository:
```sh
git clone https://github.com/n11west11/ai-assisted-test-repo.git
```

2. Change to the project directory:
```sh
cd ai-assisted-test-repo
```

3. Install the dependencies:
```sh
► INSERT-TEXT
```

### 🤖 Running ai-assisted-test-repo

```sh
► INSERT-TEXT
```

### 🧪 Tests
```sh
► INSERT-TEXT
```

---


## 🛣 Project Roadmap

> - [X] `ℹ️  Task 1: Implement X`
> - [ ] `ℹ️  Task 2: Implement Y`
> - [ ] `ℹ️ ...`


---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/n11west11/ai-assisted-test-repo/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/n11west11/ai-assisted-test-repo/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/n11west11/ai-assisted-test-repo/issues)**: Submit bugs found or log feature requests for N11WEST11.

#### *Contributing Guidelines*

<details closed>
<summary>Click to expand</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a Git client.
   ```sh
   git clone <your-forked-repo-url>
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear and concise message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

## 📄 License


This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 👏 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#Top)

---

