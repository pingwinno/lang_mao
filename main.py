import asyncio
import logging
import os

from langchain.globals import set_verbose, set_debug
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
# We'll use Ollama's python client just like before for multimodal
from telethon import TelegramClient
from telethon import events
from telethon.tl.types import Message

# ── ENV ────────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ["APIKEY"]  # Telegram bot token
API_ID = int(os.environ.get("API_ID", "0"))  # required by Telethon
API_HASH = os.environ.get("API_HASH", "")  # required by Telethon

LLM_ENDPOINT = os.environ["LLM_ENDPOINT"]
LLM_MODEL = os.environ["LLM_MODEL"]
SYSTEM_PROMPT = os.environ["LLM_PROMPT"]
THINK_MESSAGE = os.environ["THINK_MESSAGE"]
PERSONA = os.environ["PERSONA"]
BOT_NAME = os.environ["BOT_NAME"]  # command name without leading slash
BOT_NICK = os.environ["BOT_NICK"]  # your bot @username, no leading @

# ── LOGGING ────────────────────────────────────────────────────────────────────

# ── CLIENTS ───────────────────────────────────────────────────────────────────
# Telethon requires api_id/api_hash even for bots
if not API_ID or not API_HASH:
    raise RuntimeError(
        "Please set TG_API_ID and TG_API_HASH env vars (https://my.telegram.org)."
    )

set_verbose(True)
set_debug(True)

tg = TelegramClient("bot_session", API_ID, API_HASH)

judge = ChatOllama(model=LLM_MODEL, base_url=LLM_ENDPOINT, temperature=0)  # deterministic
judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict classifier. "
     "There is only 3 types of messages:"
     "PERSONA, "
     "ASSISTANT, "
     "NOT_RELATED. "
     "Decide if the USER MESSAGE is about: "
     "topics related to {persona} = PERSONA. "
     "question about something = ASSISTANT. "
     "not related to persona or not asking assistant something = NOT_RELATED. "
     "Return only one of category described above."),
    ("human", "USER MESSAGE:\n{message}\n\nAnswer (YES/NO) only:")
])
judge_chain = (judge_prompt | judge | StrOutputParser()).with_config(verbose=True)

llm = ChatOllama(model=LLM_MODEL, base_url=LLM_ENDPOINT, temperature=0.7)

mao_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise. Message from user is in format 'Name: message'. "
               "You can get name of human from that template"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

mao_chain = mao_prompt | llm

assistant_chain = assistant_prompt | llm

# Per-session message stores (keyed by session_id)
stores = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in stores:
        stores[session_id] = InMemoryChatMessageHistory()
    return stores[session_id]


mao_chat_with_history = RunnableWithMessageHistory(
    mao_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

assistant_chat_with_history = RunnableWithMessageHistory(
    assistant_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)


@tg.on(events.NewMessage())
async def my_event_handler(event):
    message: Message = event.message
    if await is_reply_to_bot(message, tg):
        logging.info("Processing reply")
        output_message = route_reply(event.sender.first_name, message.message, await message.get_reply_message(), event.chat_id)
        await event.reply(output_message)
    else:
        logging.info("Processing message")
        output_message = route(event.sender.first_name, message.message, event.chat_id)
        if output_message:
            await event.reply(output_message)


def route(username: str, message: str, chat_id) -> str:
    formated_message = f"{username}: {message}"
    verdict = judge_chain.invoke({"persona": PERSONA, "message": formated_message}).strip()
    if verdict.startswith("PERSONA"):  # YES
        return handle_personality(formated_message, chat_id)
    elif verdict.startswith("ASSISTANT"):
        return handle_normal(formated_message, chat_id)
    else:
        logging.info("No suitable topic. Skipping")
        pass


def route_reply(username: str, message: str, previous_message: str, chat_id) -> str:
    formated_message = f"{previous_message}, {username}: {message}"
    verdict = judge_chain.invoke({"persona": PERSONA, "message": formated_message}).strip()
    if verdict.startswith("PERSONA"):  # YES
        return handle_personality(message, chat_id)
    else:
        return handle_normal(message, chat_id)


def handle_personality(message: str, chat_id) -> str:
    result = mao_chat_with_history.invoke({"input": message}, config={"configurable": {"session_id": chat_id}})
    logging.info(result)
    return getattr(result, "content", str(result))


def handle_normal(message: str, chat_id):
    result = assistant_chat_with_history.invoke({"input": message}, config={"configurable": {"session_id": chat_id}})
    logging.info(result)
    return getattr(result, "content", str(result))


async def is_reply_to_bot(msg: Message, client) -> bool:
    if not msg.is_reply:
        return False

    # Fetch the message being replied to
    replied = await msg.get_reply_message()
    if not replied:
        return False

    # Get the current bot account info
    me = await client.get_me()

    return getattr(replied, "sender_id", None) == me.id


# ── MAIN ───────────────────────────────────────────────────────────────────────
async def main():
    await tg.start(bot_token=BOT_TOKEN)
    await tg.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
