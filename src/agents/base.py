import os
import sys
import datetime
import hashlib
import logging
import json

import getpass
import os


if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = getpass.getpass("API_KEY: ")

from langchain_openai import ChatOpenAI


def get_model(**kwargs) -> ChatOpenAI:
    model = ChatOpenAI(
        model=kwargs.pop("llm_model", os.getenv("LLM_MODEL", "qwen-plus")),
        temperature=0.2,
        max_tokens=2000,
        api_key=kwargs.pop("llm_api_key", os.getenv("API_KEY")),
        base_url=kwargs.pop(
            "llm_base_url",
            os.getenv(
                "LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        ),
        **kwargs,
    )
    return model


from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessageChunk,
)
from langchain.output_parsers.json import parse_json_markdown
from typing import Iterator


class AgentBase:
    def __init__(self, **kwargs):
        self.__prompt: str = kwargs.pop("prompt", "")
        if not self.__prompt or not isinstance(self.__prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        default_name = f"Agent-{hashlib.md5(self.__prompt.encode()).hexdigest()}"
        self.__name: str = kwargs.pop("name", default_name)
        self.logger = logging.getLogger(self.__name)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        log_level = kwargs.pop("log_level", logging.INFO)
        self.logger.setLevel(log_level)
        self.__usage_logger = logging.getLogger(f"usage.{self.__name}")
        self.__usage_logger.setLevel(log_level)
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
        self.__usage_logger.addHandler(
            logging.FileHandler(f"logs/usage.{self.__name}.log")
        )

        self.model = get_model(**kwargs)

    def __invoke(self, query: str, history: list[BaseMessage] = [], **kwargs) -> str:
        today = str(datetime.datetime.now().date())
        system_msg = SystemMessage(
            self.__prompt.format(
                today=today,
                **kwargs,
            ),
        )
        messages = [system_msg] + history + [HumanMessage(query)]
        self.logger.debug(f"Agent-{self.__name} __invoke messages: {messages}")
        if kwargs.get("stream", False):
            return self.model.stream(messages)
        return self.model.invoke(messages)

    def __log_usage(self, msg: BaseMessage, **kwargs):
        try:
            data = {**msg.response_metadata["token_usage"]}
            data["time"] = str(datetime.datetime.now())
            data["agent"] = self.__name
            data["model_name"] = msg.response_metadata.get("model_name", "unknown")
            self.__usage_logger.info(json.dumps(data))
        except:
            pass

    def invoke(self, query: str, history: list[BaseMessage] = [], **kwargs) -> str:
        msg: BaseMessage = self.__invoke(query, history, **kwargs)
        self.logger.debug(f"Agent-{self.__name} invoke return msg: {msg}")
        self.__log_usage(msg)
        return msg.content

    def invoke_json(
        self,
        query: str,
        history: list[BaseMessage] = [],
        retry_count: int = 1,
        **kwargs,
    ) -> dict[str, any]:
        count = 0
        while count < retry_count:
            try:
                msg: BaseMessage = self.__invoke(query, history, **kwargs)
                self.logger.debug(f"Agent-{self.__name} invoke_json return msg: {msg}")
                self.__log_usage(msg)
                parsed = parse_json_markdown(msg.content)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    # the response is not a mapping
                    return {}
            except Exception as e:
                self.logger.error(f"Agent-{self.__name} invoke_json error: {e}")
            finally:
                count += 1
        return {}

    def stream(
        self,
        query: str,
        history: list[BaseMessage] = [],
        **kwargs,
    ) -> Iterator[BaseMessageChunk]:
        return self.__invoke(query, history, stream=True, **kwargs)
