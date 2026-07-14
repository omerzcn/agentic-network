from policy_engine import DeterministicPolicyEngine

class RoutingAgent:
    def __init__(self, ctrl):
        self.policy = DeterministicPolicyEngine(ctrl)

    def route_flows(self, demands):
        return self.policy.route_flows(demands)
