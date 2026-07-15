from policy_engine import LLMPolicyEngine

class RoutingAgent:
    def __init__(self, ctrl):
        self.policy = LLMPolicyEngine(ctrl)

    def route_flows(self, demands):
        return self.policy.route_flows(demands)
    
    def get_stats(self):
        return self.policy.get_stats()
