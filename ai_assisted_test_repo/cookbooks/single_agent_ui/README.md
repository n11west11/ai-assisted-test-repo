# Qbert - A Base Chainlit Application

Qbert is a Chainlit application designed with extensibility and scalability in mind. It comes with built-in authentication and a well-organized structure that allows for easy extension of available chat profiles.

## Structure

The application is organized into several directories:

- [`ai_assisted_test_repo`](ai_assisted_test_repo): This is the main directory of the application. It contains the core logic and functionalities of the application.

- [`tools`](tools): This directory contains various utility scripts and modules that are used throughout the application. It includes modules for AWS SSM, default toolkit, embeddings, export model, Jira, test management, and UI.

- [`cookbooks`](ai_assisted_test_repo/cookbooks): This directory contains various "cookbooks" or modules that extend the functionality of the application. It includes modules for API assistant, GraphQL assistant, GraphQL vector search, HTML chain, and more.

- [`scripts`](ai_assisted_test_repo/scripts): This directory contains various scripts that are used for tasks such as building, testing, and deploying the application.

## Authentication

Qbert comes with built-in authentication. This ensures that only authorized users can access certain parts of the application.

## Extending Chat Profiles

One of the key features of Qbert is its ability to easily extend the available chat profiles. This is made possible by the well-organized structure of the application. To add a new chat profile, simply create a new module in the `profiles` directory.

## Getting Started

To get started with Qubert, clone the repository and install the necessary dependencies. Then, you can start the application and begin extending it to suit your needs.

## License

Qubert is licensed under the terms of the [LICENSE](LICENSE) file included in the repository.