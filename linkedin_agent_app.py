import os
import uuid
import asyncio

from dotenv import load_dotenv
import requests
from pydantic import BaseModel
from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    FunctionTool,
    function_tool,
    trace,
)

from linkedin_api_util import auth, get_headers, get_user_info


class LinkedInResponse(BaseModel):
    response: dict
    post_url: str


USER_NAME = "Hemantha"
load_dotenv(override=True)


@function_tool
async def linkedin_posting_tool(text_body: str) -> LinkedInResponse:
    """This tool allows you to post text to LinkedIn. The text to post is provided in the text_body argument.

    Args:
        text_body (str): The text to post to LinkedIn.
    Returns:
        LinkedInResponse: A response object with the response from LinkedIn's API and the URL of the post.
    """

    # get access token if not already available
    credentials = "credentials.json"
    access_token = auth(credentials)

    headers = get_headers(access_token=access_token)

    # Get user id to make a UGC post
    user_info = get_user_info(headers)

    text_request = {
        "author": "urn:li:person:" + user_info["sub"],
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text_body},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    response = requests.post(
        url="https://api.linkedin.com/v2/ugcPosts", headers=headers, json=text_request
    )

    post_url = "https://www.linkedin.com/feed/update/" + response.headers.get(
        "X-RestLi-Id"
    )

    linkedin_response = LinkedInResponse(response=response.json(), post_url=post_url)

    return linkedin_response


instructions = f"""
    You are Link, an AI agent created to assist {USER_NAME}, the user, in posting engaging content to LinkedIn. Your overall objective is to make it easy for the user to create and post engaging and cohesive content on LinkedIn. {USER_NAME}, the user, will initiate the conversation by briefly describing the topic they want to post on. Perform the following tasks:
    1. Interview the user to gather more information about what they want to post. Probe the user until you have enough information to create an engaging post that is exhaustive but not too long.
    2. Come up with a draft post based on the gathered information and ask the user for feedback on it.
    3. Finalize the draft based on the user's feedback and ask the user if they would like to post it.
    4. If the user agrees, post the content to LinkedIn using the LinkedIn posting tool and share the post URL with the user.

    Important Note: You should write the post from your (Link the AI agent's) perspective, as if you are the one posting it on LinkedIn using {USER_NAME}'s inputs, not {USER_NAME}. For example, 'Hi, I am Link, an AI agent...'. Your post should be engaging but not too corny or overly promotional. Keep it professional and interesting.
"""

link_ai = Agent(
    name="LinkedIn AI Assistant",
    model=os.getenv("OPENAI_MODEL"),
    instructions=instructions,
    tools=[linkedin_posting_tool],
)


async def main():
    conversation_id = uuid.uuid4().hex[:16]
    input_items = []
    while True:
        user_input = input("User: ")
        with trace("Link AI Post", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(starting_agent=link_ai, input=input_items)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")
                else:
                    print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
            input_items = result.to_input_list()


if __name__ == "__main__":
    asyncio.run(main())
