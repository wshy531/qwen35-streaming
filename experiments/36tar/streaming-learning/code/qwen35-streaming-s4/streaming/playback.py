"""PTS-paced file input and a single newest-frame slot; model state has one owner."""
import hashlib
import json
import math
import threading
import time


class SourceDigest:
    """Fingerprint every selected source frame, including frames later dropped."""
    def __init__(self):
        self.hash = hashlib.sha256()
        self.count = 0

    def add(self, packet):
        item = {"frame_id":packet.frame_id,"timestamp":packet.timestamp,"decoded_count":packet.decoded_count,
                "shape":list(packet.rgb.shape),"rgb_sha256":hashlib.sha256(packet.rgb.tobytes()).hexdigest()}
        self.hash.update((json.dumps(item,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8"))
        self.count += 1

    def snapshot(self):
        return {"samples":self.count,"sha256":self.hash.hexdigest()}


class LatestMailbox:
    """Overwrite unclaimed work, never an in-flight frame owned by the consumer."""
    def __init__(self):
        self.condition = threading.Condition()
        self.latest = None
        self.done = False
        self.error = None
        self.produced = self.delivered = self.dropped = self.cancelled = 0
        self.peak_bytes = 0

    def put(self, packet):
        with self.condition:
            if self.done:
                raise RuntimeError("Mailbox already finished")
            discarded = self.latest
            self.latest = packet
            self.produced += 1
            self.dropped += discarded is not None
            self.peak_bytes = max(self.peak_bytes,packet.rgb.nbytes)
            self.condition.notify_all()
            return None if discarded is None else discarded.frame_id

    def take(self):
        with self.condition:
            while self.latest is None and not self.done:
                self.condition.wait(timeout=0.1)
            if self.latest is not None:
                value,self.latest = self.latest,None
                self.delivered += 1
                return value
            if self.error is not None:
                raise RuntimeError("Source producer failed: "+self.error)
            raise StopIteration

    def finish(self, error=None):
        with self.condition:
            self.done = True
            self.error = error
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.cancelled += self.latest is not None
            self.latest = None
            self.done = True
            self.condition.notify_all()

    def status(self):
        with self.condition:
            buffered = self.latest is not None
            result = {"produced_samples":self.produced,"delivered_samples":self.delivered,
                      "dropped_samples":self.dropped,"cancelled_samples":self.cancelled,
                      "buffered_samples":int(buffered),"max_buffered_samples":int(self.produced > 0),
                      "buffered_rgb_bytes":0 if not buffered else self.latest.rgb.nbytes,
                      "peak_buffered_rgb_bytes":self.peak_bytes,"finished":self.done}
            assert self.produced == self.delivered+self.dropped+self.cancelled+int(buffered)
            return result


class Pacer:
    def __init__(self, stop, origin, clock=time.monotonic, wait=None):
        self.stop,self.origin,self.clock = stop,origin,clock
        self.wait = stop.wait if wait is None else wait

    def until(self, pts):
        if not math.isfinite(pts) or pts < 0:
            raise ValueError("Finite nonnegative PTS required")
        while not self.stop.is_set():
            remaining = self.origin+pts-self.clock()
            if remaining <= 0:
                return True
            self.wait(min(remaining,0.05))
        return False


class ClockPlayback:
    """The producer owns decoding; the caller alone owns GPU/session operations."""
    def __init__(self, reader, duration, max_samples, on_frame=None, clock=time.monotonic):
        if not math.isfinite(duration) or duration <= 0 or not isinstance(max_samples,int) or isinstance(max_samples,bool) or max_samples < 1:
            raise ValueError("Positive finite duration and sample budget required")
        self.reader,self.duration,self.max_samples,self.on_frame,self.clock = reader,duration,max_samples,on_frame,clock
        self.mailbox,self.digest = LatestMailbox(),SourceDigest()
        self.stop = threading.Event()
        self.origin = None
        self.thread = None
        self.reason = None
        self.max_source_lateness = 0.0

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Playback already started")
        self.origin = self.clock()
        self.thread = threading.Thread(target=self._produce,name="video-pts-reader",daemon=True)
        self.thread.start()
        return self

    def _produce(self):
        error = None
        pacer = Pacer(self.stop,self.origin,self.clock)
        try:
            self.reason = "source_eof"
            for packet in self.reader:
                if packet.timestamp >= self.duration:
                    self.reason = "video_duration_limit"
                    break
                if not pacer.until(packet.timestamp):
                    self.reason = "cancelled"
                    break
                self.digest.add(packet)
                discarded = self.mailbox.put(packet)
                wall = self.clock()-self.origin
                self.max_source_lateness = max(self.max_source_lateness,max(0.,wall-packet.timestamp))
                if self.on_frame is not None:
                    self.on_frame(packet,{"wall_seconds":wall,"source_lateness_seconds":max(0.,wall-packet.timestamp),
                                          "discarded_frame_id":discarded,**self.mailbox.status()})
                if self.digest.count >= self.max_samples:
                    self.reason = "frame_budget_limit"
                    break
        except Exception as problem:
            error = type(problem).__name__+": "+str(problem)
            self.reason = "source_error"
        finally:
            try:
                self.reader.close()
            except Exception as problem:
                error = error or "Reader close failed: "+str(problem)
            self.mailbox.finish(error)

    def __iter__(self):
        return self

    def __next__(self):
        return self.mailbox.take()

    def status(self):
        return {**self.mailbox.status(),"source_fingerprint":self.digest.snapshot(),"end_reason":self.reason,
                "max_source_lateness_seconds":self.max_source_lateness,
                "producer_alive":self.thread is not None and self.thread.is_alive(),"queue_capacity":1}

    def close(self):
        self.stop.set()
        self.mailbox.cancel()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError("Decoder did not stop; child watchdog must terminate it")
