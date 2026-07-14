class RoutingAgent:
    def __init__(self, ctrl):
        self.ctrl = ctrl

    def route_flows(self, demands):
        raise NotImplementedError(
            "Agentic routing is not implemented in Stage 1."
        )
