#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 事件驱动引擎
# 事件驱动型交易程序的核心

# Author: Tayii
# Data : 2020/12/01
# ----------------------------------------------------

from collections import defaultdict
from threading import Thread
from queue import Empty, Queue
import time
from datetime import datetime
from typing import Any, Callable, List

EVENT_TIMER = "event_timer"


class Event:
    """
    Event object consists of a type string which is used
    by engine engine for distributing engine, and a data
    object which contains the real data.
    """

    def __init__(self, type_: str, data: Any = None):
        """"""
        self.type: str = type_
        self.data: Any = data


# Defines handler function to be used in engine engine.
HandlerType = Callable[[Event], None]


class EventEngine(Thread):
    """
    Event engine distributes engine object based on its type
    to those handlers registered.

    It also generates timer engine by every interval seconds,
    which can be used for timing purpose.
    """

    def __init__(self, q: Queue = None, interval: int = 1):
        """
        Timer engine is generated every 1 second by default, if
        interval not specified.
        """
        Thread.__init__(self)
        self.name = 'Event Engine'
        self._interval: int = interval
        self._queue = q or Queue()
        self._active: bool = False
        self._timer = Thread(target=self._run_timer)
        self._handlers = defaultdict(list)
        self._general_handlers = []

    def run(self) -> None:
        """
        Start engine engine to process events and generate timer events.
        """
        self._timer.start()
        self._active = True
        print(f'{self.name} 线程启动')

        # Get engine from queue and then process it.
        while self._active:
            time.sleep(1)
            try:
                # print('EventEngine waiting......', datetime.now())
                event = self._queue.get(block=True, timeout=1)
                self._process(event)
            except Empty:
                pass

    def _process(self, event: Event) -> None:
        """
        First distribute engine to those handlers registered listening
        to this type.

        Then distribute engine to those general handlers which listens
        to all types.
        """
        if event.type in self._handlers:
            [handler(event) for handler in self._handlers[event.type]]

        if self._general_handlers:
            [handler(event) for handler in self._general_handlers]

    def _run_timer(self) -> None:
        """
        Sleep by interval second(s) and then generate a timer engine.
        """
        while self._active:
            time.sleep(self._interval)
            event = Event(EVENT_TIMER)
            self.put(event)

    def stop(self) -> None:
        """
        Stop engine engine.
        """
        self._active = False
        self._timer.join()

    def put(self, event: Event) -> None:
        """
        Put an engine object into engine queue.
        """
        self._queue.put(event)

    def register(self, type: str, handler: HandlerType) -> None:
        """
        Register a new handler function for a specific engine type. Every
        function can only be registered once for each engine type.
        """
        handler_list = self._handlers[type]
        if handler not in handler_list:
            handler_list.append(handler)

    def unregister(self, type: str, handler: HandlerType) -> None:
        """
        Unregister an existing handler function from engine engine.
        """
        handler_list = self._handlers[type]

        if handler in handler_list:
            handler_list.remove(handler)

        if not handler_list:
            self._handlers.pop(type)

    def register_general(self, handler: HandlerType) -> None:
        """
        Register a new handler function for all engine types. Every
        function can only be registered once for each engine type.
        """
        if handler not in self._general_handlers:
            print(f'{handler} register_general')
            self._general_handlers.append(handler)

    def unregister_general(self, handler: HandlerType) -> None:
        """
        Unregister an existing general handler function.
        """
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)
