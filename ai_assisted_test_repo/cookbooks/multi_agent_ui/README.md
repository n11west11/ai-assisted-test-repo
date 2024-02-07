# Qubert Chainlit Application

Qbert is a UI test assistant. It is designed to help interact with the UI of a web application and create test cases for the application. Qubert is built on top of the [Chainlit](
command:_github.copilot.openRelativePath?%5B%22../chainlit/chainlit.md%22%5D
 "Chainlit") conversational retrieval agent.


## Structure

The Qubert application is organized into several key components:

- `chainlit`: The Front end that powers Qubert. It is primarily run by calling the app.py file.
- `profiles`: The font end is intended to be divided into different profiles. This application only has one profile, `multi_agent_ui`, which is designed to interact with the UI of a web application and create test cases for the application. More profiles can be added in the future.
- `tools`: The tools directory contains a set of tools that can be used to interact with the Qubert application. These are the core tools that each profile will use to interact with the application. Within the `tools` directory, there are toolkits which are indented to split the tools to their corresponding agents. More info on agents can be found below in the `Agents` section.

## Agents
In order for Qubert to properly detect which tools to use, it is split into agents. Each agent has a different set of tools that it can use. The agents are as follows:
- `test execution agent`: This agent is responsible for executing the tests that are created by the user. It is also responsible for saving the steps that the user takes in order to create the test cases.
- `test documentation agent`: This agent is responsible for creating the test cases that the user wants to create. It is also responsible for saving the steps that the user takes in order to create the test cases.



## Extending Chat Profiles

Qubert is designed to be easily extendable. To add a new chat profile, create a new module in the `profiles` directory. The module should be built from the base profile class
and implement the methods for initialize, and process_message.

## Testing



## Getting Started

To get started with Qubert, follow the instructions in the [`README.md`](command:_github.copilot.openRelativePath?%5B%22README.md%22%5D "README.md") file. This includes instructions on cloning the repository, installing dependencies, and starting the application.

## License

Qubert is licensed under the terms of the [LICENSE](command:_github.copilot.openRelativePath?%5B%22LICENSE%22%5D "LICENSE") file included in the repository.