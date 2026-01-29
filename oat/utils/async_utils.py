# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async utilities for OAT."""

import asyncio
import threading


class AsyncLoopThread:
    """Run a persistent event loop in a background thread for AsyncLLMEngine.

    AsyncLLMEngine in vLLM v1 requires a persistent event loop to process requests.
    This class creates a background thread with a running event loop and provides
    methods to submit coroutines to it from sync code.

    Example:
        loop_thread = AsyncLoopThread()
        loop_thread.start()

        # Submit coroutines from sync code
        result = loop_thread.run_coroutine(some_async_function())

        # Clean up when done
        loop_thread.stop()
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._started = threading.Event()

    def start(self):
        """Start the background thread with an event loop."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait for loop to be running
        self._started.wait(timeout=10)

    def _run_loop(self):
        """Internal method to run the event loop in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run_coroutine(self, coro):
        """Submit a coroutine to the event loop and wait for result.

        Args:
            coro: A coroutine object to run.

        Returns:
            The result of the coroutine.

        Raises:
            RuntimeError: If the event loop is not started or not running.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("Event loop not started or not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def run_gather(self, *coros):
        """Run multiple coroutines concurrently and wait for all results.

        This method handles asyncio.gather inside the event loop thread,
        avoiding "no current event loop" errors when called from sync code.

        Args:
            *coros: Coroutine objects to run concurrently.

        Returns:
            List of results from all coroutines.
        """
        async def _gather():
            return await asyncio.gather(*coros)
        return self.run_coroutine(_gather())

    def stop(self):
        """Stop the event loop and join the background thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def loop(self):
        """Return the underlying event loop."""
        return self._loop

    @property
    def is_running(self):
        """Check if the event loop is running."""
        return self._loop is not None and self._loop.is_running()
