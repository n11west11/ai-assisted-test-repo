from typing import Optional
from ai_assisted_test_repo.cookbooks.multi_agent_ui.profiles.base_profile import BaseProfile
from ai_assisted_test_repo.cookbooks.multi_agent_ui.profiles.ui.ui_app import UITestProfile

import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider, Tags
from openai import chat

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

@cl.set_chat_profiles
async def chat_profile():
    return [
        UITestProfile.chat_profile,
    ]

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
  # Fetch the user matching username from your database
  # and compare the hashed password with the value stored in the database
  if (username, password) == ("admin", "admin"):
    return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
  else:
    return None

@cl.on_chat_start
async def start():
    chat_profile = cl.user_session.get("chat_profile") # type: str
    match chat_profile:
        case UITestProfile.name:
            profile = UITestProfile()
            await profile.initialize()
    cl.user_session.set("profile", profile)
    await cl.Message(
        content=f"Starting chat using the {profile.name} chat profile"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    profile = cl.user_session.get("profile") # type: UITestProfile
    await profile.process_message(message)