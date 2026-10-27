import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from . import gtm_agent as agent


class GTMAgentTests(unittest.TestCase):
    def run_workflow(self, request, steps):
        responses = iter([
            AIMessage(content="", tool_calls=[
                {"name": "get_current_rep", "args": {}, "id": "rep"},
            ]),
            *[
                AIMessage(content="", tool_calls=[
                    {"name": name, "args": args, "id": str(index)},
                ])
                for index, (name, args) in enumerate(steps)
            ],
            AIMessage(content="Completed."),
        ])
        requests = []

        def generate(model, messages, **kwargs):
            requests.append({"messages": messages_to_dict(messages), "tools": kwargs["tools"]})
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
                patch.object(ChatOpenAI, "_generate", generate):
            result = agent.run_agent(request, user_id=agent.REP_IDS[0]["rep_id"])

        self.assertEqual(result["reply"], "Completed.")
        outputs = [json.loads(message.content) for message in result["messages"]
                   if isinstance(message, ToolMessage)]
        self.assertEqual(outputs[0]["rep"], agent.REP_IDS[0])
        first_request = requests[0]
        self.assertEqual({tool["function"]["name"] for tool in first_request["tools"]}, {
            "get_current_rep", "get_prospect", "send_prospect_email",
            "build_prospect_profile", "lookup_offering", "score_prospect",
            "update_prospect_info",
        })
        request_bytes = len(json.dumps(first_request).encode("utf-8"))
        self.assertLess(request_bytes, 4500)
        print(f"{self._testMethodName}: first request serialized bytes = {request_bytes}")
        return outputs

    def test_scoring_workflow(self):
        prospect_id = "LEAD-12853"
        with patch.dict(agent.data_service._PROFILES, clear=True):
            profile = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
            agent.data_service._PROFILES.clear()
            offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10003"})["offering"]
            score = agent.ProspectScore(
                score=80, justification="Good fit", rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=90, tech_stack_match=80, segment_fit=70,
                ),
            )
            with patch.object(agent, "_scoring_llm") as scoring_llm:
                scoring_llm.invoke.return_value = score
                outputs = self.run_workflow(
                    "Score LEAD-12853 for OFFER-10003.",
                    [("build_prospect_profile", {"prospect_id": prospect_id}),
                     ("lookup_offering", {"offering_id": "OFFER-10003"}),
                     ("score_prospect", {"prospect_profile": profile, "offering": offering})],
                )
            self.assertEqual(outputs[1]["prospect_profile"], profile)
            self.assertEqual(outputs[2]["offering"], offering)
            self.assertEqual(outputs[3], score.model_dump())

    def test_email_workflow(self):
        prospect_id = "LEAD-12853"
        prospect = agent.get_prospect.invoke({"prospect_id": prospect_id})["prospect"]
        outputs = self.run_workflow(
            "Email LEAD-12853 to arrange a demo.",
            [("get_prospect", {"prospect_id": prospect_id}),
             ("send_prospect_email", {"prospect": prospect, "subject": "Demo", "body": "Let's arrange a demo."})],
        )
        self.assertEqual(outputs[1]["prospect"], prospect)
        self.assertEqual(outputs[2]["status"], "sent")
        self.assertEqual(outputs[2]["to"], prospect["email"])
        self.assertEqual(outputs[2]["from"], agent.REP_IDS[0]["email"])

    def test_update_workflow(self):
        outputs = self.run_workflow(
            "Add Rust to LEAD-12853's technology stack.",
            [("update_prospect_info", {"prospect_id": "LEAD-12853", "technology": "Rust"})],
        )
        self.assertTrue(outputs[1]["updated"])
        self.assertTrue(outputs[1]["found"])
        self.assertIn("Rust", outputs[1]["tech_stack"])


if __name__ == "__main__":
    unittest.main()
