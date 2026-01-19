import datetime
import hashlib
import json
from typing import Any, Iterator

from langchain_core.utils.json import parse_json_markdown
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from src.common.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_COUNT,
    DEFAULT_TEMPERATURE,
    get_api_key,
    get_llm_base_url,
    get_llm_model,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


# Initialize API key if not set
get_api_key()


def get_model(**kwargs) -> ChatOpenAI:
    """
    Create and configure a ChatOpenAI model instance.

    Args:
        **kwargs: Configuration parameters including:
            - llm_model: Model name (default: from env or DEFAULT_LLM_MODEL)
            - llm_api_key: API key (default: from env)
            - llm_base_url: Base URL (default: from env or DEFAULT_BASE_URL)
            - temperature: Temperature setting (default: DEFAULT_TEMPERATURE)
            - max_tokens: Maximum tokens (default: DEFAULT_MAX_TOKENS)
            - Other ChatOpenAI parameters

    Returns:
        Configured ChatOpenAI instance
    """
    llm_model = kwargs.pop("llm_model", get_llm_model())
    temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
    max_tokens = kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
    base_url = kwargs.pop("llm_base_url", get_llm_base_url())
    
    logger.info(f"Creating ChatOpenAI model: model={llm_model}, temperature={temperature}, max_tokens={max_tokens}, base_url={base_url}")
    
    model = ChatOpenAI(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=kwargs.pop("llm_api_key", get_api_key()),
        base_url=base_url,
        **kwargs,
    )
    logger.debug("ChatOpenAI model created successfully")
    return model


class AgentBase:
    """
    Base class for all agents in the system.

    Provides common functionality for:
    - LLM model initialization and configuration
    - Message invocation (sync, async, JSON parsing, streaming)
    """

    def __init__(self, **kwargs):
        """
        Initialize the agent.

        Args:
            **kwargs: Configuration parameters including:
                - prompt: System prompt string (required)
                - name: Agent name (optional, auto-generated if not provided)
                - llm_model: LLM model name
                - llm_api_key: API key
                - llm_base_url: Base URL
                - Other model configuration parameters
        """
        # Validate and set prompt
        self.__prompt: str = kwargs.pop("prompt", "")
        if not self.__prompt or not isinstance(self.__prompt, str):
            logger.error("Prompt must be a non-empty string")
            raise ValueError("Prompt must be a non-empty string")

        # Generate default name from prompt hash if not provided
        default_name = f"Agent-{hashlib.md5(self.__prompt.encode()).hexdigest()}"
        self.__name: str = kwargs.pop("name", default_name)
        logger.info(f"Initializing AgentBase: name={self.__name}")

        # Initialize model
        self.model = get_model(**kwargs)
        logger.info(f"AgentBase initialized successfully: {self.__name}")

    def __invoke(self, query: str, history: list[BaseMessage] = [], **kwargs):
        """
        Internal method to invoke the model with formatted messages.

        Args:
            query: User query string
            history: List of previous messages for context
            **kwargs: Additional parameters for prompt formatting and model invocation

        Returns:
            Model response (BaseMessage or Iterator depending on stream parameter)
        """
        today = str(datetime.datetime.now().date())
        system_msg = SystemMessage(
            self.__prompt.format(
                today=today,
                **kwargs,
            ),
        )
        messages = [system_msg] + history + [HumanMessage(query)]
        logger.debug(f"Invoking model for agent {self.__name}, query length: {len(query)}, history length: {len(history)}")

        if kwargs.get("stream", False):
            logger.debug(f"Streaming response for agent {self.__name}")
            return self.model.stream(messages)
        logger.debug(f"Invoking synchronous response for agent {self.__name}")
        return self.model.invoke(messages)

    def __log_usage(self, msg: BaseMessage, **kwargs):
        """
        Log token usage information.

        Args:
            msg: Response message containing usage metadata
            **kwargs: Additional logging parameters
        """
        try:
            data = {**msg.response_metadata["token_usage"]}
            data["time"] = str(datetime.datetime.now())
            data["agent"] = self.__name
            data["model_name"] = msg.response_metadata.get("model_name", "unknown")
            logger.info(f"Token usage: {json.dumps(data)}")
        except Exception:
            # Silently ignore logging errors
            pass

    def invoke(self, query: str, history: list[BaseMessage] = [], **kwargs) -> str:
        """
        Invoke the model synchronously and return the response content.

        Args:
            query: User query string
            history: List of previous messages for context
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Response content as string
        """
        logger.info(f"Invoking agent {self.__name} with query: {query[:100]}...")
        msg: BaseMessage = self.__invoke(query, history, **kwargs)
        self.__log_usage(msg)
        logger.debug(f"Agent {self.__name} invoke completed, response length: {len(msg.content)}")
        return msg.content

    def invoke_json(
        self,
        query: str,
        history: list[BaseMessage] = [],
        retry_count: int = DEFAULT_RETRY_COUNT,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Invoke the model and parse the response as JSON.

        Args:
            query: User query string
            history: List of previous messages for context
            retry_count: Number of retry attempts if parsing fails
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Parsed JSON response as dictionary, empty dict on failure
        """
        logger.info(f"Invoking agent {self.__name} with JSON parsing, retry_count={retry_count}")
        count = 0
        while count < retry_count:
            try:
                msg: BaseMessage = self.__invoke(query, history, **kwargs)
                self.__log_usage(msg)
                parsed = parse_json_markdown(msg.content)
                if isinstance(parsed, dict):
                    logger.debug(f"Agent {self.__name} JSON parsing successful")
                    return parsed
                else:
                    # Response is not a mapping
                    logger.warning(f"Agent {self.__name} JSON parsing returned non-dict type: {type(parsed)}")
                    return {}
            except Exception as e:
                logger.error(f"Agent {self.__name} invoke_json error on attempt {count + 1}/{retry_count}: {e}")
            finally:
                count += 1
        logger.warning(f"Agent {self.__name} invoke_json failed after {retry_count} attempts")
        return {}

    def stream(
        self,
        query: str,
        history: list[BaseMessage] = [],
        **kwargs,
    ) -> Iterator[BaseMessageChunk]:
        """
        Stream the model response as an iterator of message chunks.

        Args:
            query: User query string
            history: List of previous messages for context
            **kwargs: Additional parameters for prompt formatting

        Returns:
            Iterator of BaseMessageChunk objects
        """
        logger.info(f"Streaming response for agent {self.__name}")
        return self.__invoke(query, history, stream=True, **kwargs)
