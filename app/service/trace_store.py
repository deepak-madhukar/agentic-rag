from typing import Optional


class TraceStore:
    def __init__(self):
        self.traces: dict[str, dict] = {}

    def store(self, request_id: str, trace: dict) -> None:
        self.traces[request_id] = trace

    def get(self, request_id: str) -> Optional[dict]:
        return self.traces.get(request_id)

    def get_latest(self) -> Optional[dict]:
        if not self.traces:
            return None

        latest_id = max(self.traces.keys(), key=lambda x: self.traces[x].get("timestamp", 0))
        return self.traces[latest_id]


_trace_store = TraceStore()


def get_trace_store() -> TraceStore:
    return _trace_store