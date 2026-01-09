"""
SignalBus - Thread-safe pub/sub system for signal distribution.

Provides:
- Non-blocking publish() for producers (safe from audio callback)
- Per-subscriber queues with configurable size
- Background dispatcher thread for fan-out
- Type-based and source-based filtering
"""

import queue
import threading
import time
from collections import defaultdict
from typing import Callable, Optional, Type

from lib.signals.base import AnySignal, AudioSignal, ScalarSignal, Signal, TextSignal


class Subscription:
    """
    Represents a subscription to the signal bus.
    
    Each subscriber gets their own queue to prevent slow consumers
    from blocking the dispatch thread.
    """
    
    def __init__(
        self,
        callback: Callable[[AnySignal], None],
        signal_type: Optional[Type[Signal]] = None,
        source_filter: Optional[str] = None,
        queue_size: int = 100
    ):
        """
        Create a subscription.
        
        Args:
            callback: Function to call with each signal
            signal_type: Only receive signals of this type (None = all)
            source_filter: Only receive signals from this source (None = all)
            queue_size: Max pending signals before dropping
        """
        self.callback = callback
        self.signal_type = signal_type
        self.source_filter = source_filter
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._active = True
        
        # Stats
        self.received_count = 0
        self.dropped_count = 0
    
    def matches(self, signal: Signal) -> bool:
        """Check if this subscription should receive the signal."""
        if not self._active:
            return False
        
        # Type filter
        if self.signal_type is not None:
            if not isinstance(signal, self.signal_type):
                return False
        
        # Source filter
        if self.source_filter is not None:
            if signal.source != self.source_filter:
                return False
        
        return True
    
    def enqueue(self, signal: Signal) -> bool:
        """
        Add signal to subscriber's queue.
        
        Returns:
            True if enqueued, False if dropped (queue full)
        """
        try:
            self.queue.put_nowait(signal)
            self.received_count += 1
            return True
        except queue.Full:
            self.dropped_count += 1
            return False
    
    def deactivate(self):
        """Mark subscription as inactive (pending removal)."""
        self._active = False


class SignalBus:
    """
    Thread-safe signal distribution bus.
    
    Producers call publish() which is non-blocking and safe from any thread.
    Subscribers receive signals via callbacks in the dispatcher thread.
    
    Usage:
        bus = SignalBus()
        bus.start()
        
        # Subscribe to all TextSignals
        bus.subscribe(TextSignal, lambda s: print(s.message))
        
        # Publish from anywhere
        bus.publish(TextSignal(category="test", message="Hello"))
        
        bus.stop()
    """
    
    def __init__(self, dispatch_queue_size: int = 1000):
        """
        Create a signal bus.
        
        Args:
            dispatch_queue_size: Size of main dispatch queue
        """
        self._dispatch_queue: queue.Queue = queue.Queue(maxsize=dispatch_queue_size)
        self._subscriptions: list[Subscription] = []
        self._subscriptions_lock = threading.Lock()
        
        # Dispatcher thread
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        # Stats
        self._published_count = 0
        self._dispatched_count = 0
        self._dropped_count = 0
    
    @property
    def is_running(self) -> bool:
        """Check if dispatcher is running."""
        return self._running
    
    def start(self) -> None:
        """Start the dispatcher thread."""
        if self._running:
            return
        
        self._stop_event.clear()
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="SignalBusDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()
        self._running = True
    
    def stop(self, timeout: float = 2.0) -> None:
        """
        Stop the dispatcher thread.
        
        Args:
            timeout: Max seconds to wait for thread to stop
        """
        if not self._running:
            return
        
        self._stop_event.set()
        
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=timeout)
        
        self._dispatcher_thread = None
        self._running = False
    
    def publish(self, signal: Signal) -> bool:
        """
        Publish a signal (non-blocking).
        
        Safe to call from any thread, including audio callbacks.
        
        Args:
            signal: Signal to publish
            
        Returns:
            True if queued, False if dropped (queue full)
        """
        try:
            self._dispatch_queue.put_nowait(signal)
            self._published_count += 1
            return True
        except queue.Full:
            self._dropped_count += 1
            return False
    
    def subscribe(
        self,
        signal_type: Optional[Type[Signal]] = None,
        callback: Optional[Callable[[AnySignal], None]] = None,
        source_filter: Optional[str] = None,
        queue_size: int = 100
    ) -> Subscription:
        """
        Subscribe to signals.
        
        Args:
            signal_type: Type of signal to receive (None = all)
            callback: Function to call with each signal
            source_filter: Only receive from this source (None = all)
            queue_size: Max pending signals before dropping
            
        Returns:
            Subscription object (can be used to unsubscribe)
        """
        if callback is None:
            raise ValueError("callback is required")
        
        subscription = Subscription(
            callback=callback,
            signal_type=signal_type,
            source_filter=source_filter,
            queue_size=queue_size
        )
        
        with self._subscriptions_lock:
            self._subscriptions.append(subscription)
        
        return subscription
    
    def unsubscribe(self, subscription: Subscription) -> bool:
        """
        Remove a subscription.
        
        Args:
            subscription: Subscription to remove
            
        Returns:
            True if removed, False if not found
        """
        subscription.deactivate()
        
        with self._subscriptions_lock:
            try:
                self._subscriptions.remove(subscription)
                return True
            except ValueError:
                return False
    
    def _dispatcher_loop(self) -> None:
        """
        Main dispatch loop (runs in background thread).
        
        Pulls signals from dispatch queue and fans out to subscribers.
        """
        while not self._stop_event.is_set():
            try:
                # Get signal with timeout (allows checking stop event)
                try:
                    signal = self._dispatch_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Fan out to matching subscribers
                with self._subscriptions_lock:
                    subscriptions = list(self._subscriptions)
                
                for sub in subscriptions:
                    if sub.matches(signal):
                        if sub.enqueue(signal):
                            # Call callback (in dispatcher thread)
                            try:
                                sub.callback(signal)
                            except Exception as e:
                                # Log but don't crash dispatcher
                                print(f"⚠️  SignalBus callback error: {e}")
                
                self._dispatched_count += 1
                
            except Exception as e:
                # Log but continue
                print(f"⚠️  SignalBus dispatch error: {e}")
                time.sleep(0.01)
    
    def get_stats(self) -> dict:
        """Get bus statistics."""
        with self._subscriptions_lock:
            sub_stats = [
                {
                    "type": sub.signal_type.__name__ if sub.signal_type else "all",
                    "source": sub.source_filter or "all",
                    "received": sub.received_count,
                    "dropped": sub.dropped_count,
                    "pending": sub.queue.qsize(),
                }
                for sub in self._subscriptions
            ]
        
        return {
            "published": self._published_count,
            "dispatched": self._dispatched_count,
            "dropped": self._dropped_count,
            "queue_size": self._dispatch_queue.qsize(),
            "subscribers": sub_stats,
        }

