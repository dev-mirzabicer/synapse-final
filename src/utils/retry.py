import functools
import logging
from typing import Callable, Any, TypeVar
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from src.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for adding retry logic to functions.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exceptions: Tuple of exceptions to retry on
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Function execution failed, will retry if attempts remain",
                    function=func.__name__,
                    error=str(e),
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys()),
                )
                raise

        return wrapper

    return decorator


def with_async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for adding retry logic to async functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Async function execution failed, will retry if attempts remain",
                    function=func.__name__,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator
