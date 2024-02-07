# Qubert Chainlit Application

Qubert is a base Chainlit application that provides a solid foundation for building chatbot applications. It comes with built-in authentication and a well-organized structure that allows for easy extension of available chat profiles.

## Structure

The Qubert application is organized into several key components:

- **UI Assistant**: This component handles the interaction with various Uis.

- **Automation Assistant**: This component component will try to answer questions about the automation team

## Extending Chat Profiles

Qubert is designed to be easily extendable. To add a new chat profile, create a new module in the `cookbooks` directory. Each module should contain a `create_conversational_retrieval_agent` function that defines the behavior of the chat profile.

## Testing

Each chat profile should be tested in its own context. The `chainlit.md` file in each `cookbooks` directory provides guidance on choosing a testing context.

## Getting Started

To get started with Qubert, follow the instructions in the [`README.md`](command:_github.copilot.openRelativePath?%5B%22README.md%22%5D "README.md") file. This includes instructions on cloning the repository, installing dependencies, and starting the application.

## License

Qubert is licensed under the terms of the [LICENSE](command:_github.copilot.openRelativePath?%5B%22LICENSE%22%5D "LICENSE") file included in the repository.